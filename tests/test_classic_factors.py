from dataclasses import replace
from decimal import Decimal
import json

import numpy as np
import pandas as pd
import pytest
import torch

from ashare_bt.classic_factors import (
    ClassicFactorStrategy, FACTORS, compute_classic_features, dividend_panel,
    financial_panels, financial_record, financial_state, ttm,
)
from ashare_bt.compute import compute_features
from ashare_bt.config import Config
from ashare_bt.data import History, MarketData
from ashare_bt.engine import run
from ashare_bt.factor_analytics import curve_metrics, rank_correlations, theoretical_curve
from ashare_bt.factor_sources import PublicDataCache, SourceQuery
from ashare_bt.status import StatusProvider


def test_ttm_and_financial_formulas_have_hand_calculated_values():
    income = {period: {"revenue": revenue, "profit": revenue / 10} for period, revenue in (
        ("2021-09-30", 70), ("2021-12-31", 100), ("2022-09-30", 90),
        ("2022-12-31", 130), ("2023-09-30", 120))}
    balance = {"2022-09-30": {"parent_equity": 80, "assets": 150},
               "2023-09-30": {"parent_equity": 120, "assets": 180}}
    assert ttm(income, "2023-09-30", "revenue") == 160
    assert ttm(income, "2022-09-30", "revenue") == 120
    assert ttm(income, "2022-12-31", "revenue") == 130
    state = financial_state(balance, income)
    assert state["book"] == 120
    assert state["quality"] == pytest.approx(.16)
    assert state["growth"] == pytest.approx(1 / 3)
    assert state["investment"] == pytest.approx(.2)
    del income["2021-12-31"]
    assert np.isnan(financial_state(balance, income)["growth"])
    balance["2022-09-30"]["parent_equity"] = -80
    assert np.isnan(financial_state(balance, income)["quality"])


def record(period, notice, update, book, source="a"):
    return financial_record({"SECUCODE": "600000.SH", "REPORT_DATE": period, "NOTICE_DATE": notice,
                             "UPDATE_DATE": update, "TOTAL_PARENT_EQUITY": book,
                             "TOTAL_ASSETS": book * 2, "CURRENCY": "CNY"}, "balance", source)


def test_financial_revision_enters_after_update_and_preserves_newer_period():
    dates = ["2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07", "2025-01-08"]
    old = record("2023-12-31", "2024-04-01", "2024-04-01", 80)
    recent = record("2024-09-30", "2025-01-02", "2025-01-06", 100)
    old_revision = record("2023-12-31", "2024-04-01", "2025-01-07", 999, "b")
    panels = financial_panels([old, recent, old_revision], dates, ["600000.SH"])
    np.testing.assert_array_equal(panels["book"][:, 0], [80, 80, 80, 100, 100])
    changed = {**recent, "parent_equity": 200}
    other = financial_panels([old, changed, old_revision], dates, ["600000.SH"])
    np.testing.assert_array_equal(panels["book"][:3], other["book"][:3])
    assert financial_record({"REPORT_DATE": "2024-09-30", "NOTICE_DATE": "2024-10-31", "CURRENCY": "CNY"}, "balance") is None


def test_dividend_units_implementation_timing_zero_and_expiry():
    dates = ["2025-01-02", "2025-01-03", "2025-01-06", "2026-01-01", "2026-01-02"]
    raw = {"SECUCODE": "600000.SH", "EX_DIVIDEND_DATE": "2025-01-02", "NOTICE_DATE": "2025-01-02",
           "REPORT_DATE": "2024-12-31", "ASSIGN_PROGRESS": "实施分配", "PRETAX_BONUS_RMB": 2, "TOTAL_SHARES": 1000}
    panel, audit = dividend_panel([raw], dates, ["600000.SH", "000001.SZ"])
    np.testing.assert_array_equal(panel[:, 0], [0, 200, 200, 200, 0])
    np.testing.assert_array_equal(panel[:, 1], np.zeros(5))
    assert audit["invalid_events"] == 0
    missing, _ = dividend_panel([raw], dates, ["600000.SH"], complete=False)
    assert np.isnan(missing).all()
    invalid, audit = dividend_panel([{**raw, "TOTAL_SHARES": None}], dates, ["600000.SH"])
    assert np.isnan(invalid[1:4]).all() and audit["invalid_events"] == 1
    plan, _ = dividend_panel([{**raw, "ASSIGN_PROGRESS": "董事会预案"}], dates, ["600000.SH"])
    assert (plan == 0).all()


def test_source_cache_checks_pagination_and_raw_hash(tmp_path, monkeypatch):
    query = SourceQuery("shares", "TEST_REPORT", "(date='2025-01-02')")
    cache = PublicDataCache(tmp_path)
    responses = [
        {"success": True, "result": {"pages": 2, "count": 3, "data": [{"id": 1}, {"id": 2}]}},
        {"success": True, "result": {"pages": 2, "count": 3, "data": [{"id": 3}]}},
    ]
    called = []
    def request(url):
        called.append(url)
        return responses[len(called) - 1]
    monkeypatch.setattr(cache, "_request", request)
    manifest = cache.fetch(query)
    assert manifest["rows"] == 3 and len(called) == 2
    assert PublicDataCache.read(manifest) == [{"id": 1}, {"id": 2}, {"id": 3}]
    assert cache.fetch(query) == manifest and len(called) == 2
    path = tmp_path / "shares" / query.key / "page-0001.json.gz"
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="哈希"):
        PublicDataCache.read(manifest)


def test_curve_risk_metrics_include_initial_cash():
    metrics = curve_metrics([1.2, .9, 1.1])
    assert metrics["return"] == pytest.approx(.1)
    assert metrics["max_drawdown"] == pytest.approx(.25)
    assert metrics["annual_return"] == pytest.approx(1.1 ** 84 - 1)
    assert np.isnan(curve_metrics([1., 1.])["calmar"])


def test_theoretical_reference_marks_overnight_and_preserves_halted_holdings():
    dates = ["2024-12-31", "2025-01-02", "2025-01-03", "2025-01-06"]
    close = np.array([[10.], [11.], [np.nan], [13.]])
    opening = np.array([[10.], [10.], [np.nan], [12.]])
    data = MarketData(dates, ["600000.SH"], {"close": close, "open": opening, "cumadj": np.ones_like(close)})
    config = Config(start=dates[1], end=dates[-1], warmup_start=dates[0], target_exposure=".95")
    curve = theoretical_curve(data, config, {0: [0]})
    np.testing.assert_allclose(curve.unit_nav, [1.095, 1.095, 1.285])


@pytest.mark.gpu
@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="使用 Apple MPS 实机运行")
def test_mps_ten_formulas_blocks_strategy_and_cache_replay(tmp_path):
    dates = pd.bdate_range("2023-01-02", periods=330).strftime("%Y-%m-%d").tolist()
    shape = len(dates), 3
    values = np.arange(len(dates))[:, None]
    close = np.exp(values * np.array([.001, .002, .003])[None, :]) * 10
    arrays = {"open": close.copy(), "close": close, "high": close * 1.01, "low": close * .99,
              "amt": np.full(shape, 1e9), "num_trades": np.full(shape, 1000.), "cumadj": np.ones(shape)}
    data = MarketData(dates, ["000001.SZ", "600000.SH", "300001.SZ"], arrays)
    config = Config(root=tmp_path, warmup_start=dates[0], start=dates[270], end=dates[-1], device="mps",
                    block_days=64, strategy="classic_value", max_positions=2, initial_cash="1000000",
                    exit_mode="strict_end", auto_sync_status=False)
    base = compute_features(data, config, use_cache=False)
    fundamentals = {"shares": np.full(shape, 100.), "book": np.full(shape, 120.),
                    "quality": np.full(shape, .16), "growth": np.full(shape, 1 / 3),
                    "investment": np.full(shape, .2), "dividend_cash": np.full(shape, 20.)}
    features, scores, common = compute_classic_features(data, base, fundamentals, config, "fixture")
    row = 280
    assert common[row].all() and not common[251].any()
    np.testing.assert_allclose(scores["value"][row], 120 / (close[row] * 100), rtol=1e-5)
    np.testing.assert_allclose(scores["size"][row], -close[row] * 100, rtol=1e-5)
    np.testing.assert_allclose(scores["momentum"][row], close[row-21] / close[row-252] - 1, rtol=1e-5)
    np.testing.assert_allclose(scores["reversal"][row], -(close[row] / close[row-5] - 1), rtol=2e-5)
    np.testing.assert_allclose(scores["quality"][row], .16, rtol=1e-6)
    np.testing.assert_allclose(scores["growth"][row], 1/3, rtol=1e-6)
    np.testing.assert_allclose(scores["investment"][row], -.2, rtol=1e-6)
    np.testing.assert_allclose(scores["dividend"][row], 20 / (close[row] * 100), rtol=1e-5)
    returns = close[1:] / close[:-1] - 1
    np.testing.assert_allclose(scores["liquidity"][row], np.abs(returns[row-20:row]).mean(axis=0) / 1e9, rtol=1e-4)
    np.testing.assert_allclose(scores["low_volatility"][row], -returns[row-60:row].std(axis=0), atol=1e-7)
    second, other, _ = compute_classic_features(data, base, fundamentals, replace(config, block_days=128), "fixture")
    for factor in FACTORS:
        np.testing.assert_allclose(scores[factor.id], other[factor.id], atol=1e-6, rtol=1e-5, equal_nan=True)
    status = StatusProvider.synthetic(dates, data.securities)
    metadata = features.metadata["classic_factors"]
    config.parameters = {"factor_cache": metadata["cache"], "factor_cache_sha256": metadata["cache_sha256"],
                         "factor_fingerprint": metadata["fingerprint"]}
    native = run(config, ClassicFactorStrategy(config, scores, dates, common), data=data, features=features, status=status, output=False)
    replay = run(config, data=data, features=base, status=status, output=False)
    pd.testing.assert_frame_equal(native.trades, replay.trades, check_exact=True)
    pd.testing.assert_frame_equal(native.equity, replay.equity, check_exact=True)
    history = History(data, features, status, row, np.array([True, False, False]), np.zeros(3, dtype=int))
    strategy = ClassicFactorStrategy(config, scores, dates, common)
    target = strategy.on_close(history, None)
    assert target.weights == {"000001.SZ": Decimal(".475000000000")}
    correlations = rank_correlations(np.array([[1, 2, 2, 4], [1, 3, 3, 4], [4, 2, 2, 1]]), torch.device("mps"))
    assert correlations[0, 1] == pytest.approx(1., abs=1e-6)
    assert correlations[0, 2] == pytest.approx(-1., abs=1e-6)
    cutoff = 300
    changed_arrays = {name: value.copy() for name, value in arrays.items()}
    for name in ("open", "high", "low", "close"):
        changed_arrays[name][cutoff:] *= 1.5
    changed_data = MarketData(dates, data.securities, changed_arrays, fingerprint="future-change")
    changed_base = compute_features(changed_data, config, use_cache=False)
    _, changed_scores, changed_common = compute_classic_features(changed_data, changed_base, fundamentals, config, "fixture")
    for factor in FACTORS:
        np.testing.assert_array_equal(scores[factor.id][:cutoff], changed_scores[factor.id][:cutoff])
    np.testing.assert_array_equal(common[:cutoff], changed_common[:cutoff])
