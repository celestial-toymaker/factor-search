"""批量准备免费数据、运行十因子研究，并验证逐笔重放。"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pyarrow.feather as feather
import pyarrow.parquet as pq

from ashare_bt.classic_factors import FACTORS, ClassicFactorStrategy, compute_classic_features, load_fundamentals
from ashare_bt.compute import FeatureBundle, compute_features, select_device
from ashare_bt.config import Config
from ashare_bt.data import load_data
from ashare_bt.engine import run, status_requirement
from ashare_bt.factor_analytics import clean, enrich_result, factor_diagnostics, write_study_report
from ashare_bt.factor_sources import fetch_sources
from ashare_bt.status import StatusProvider
from ashare_bt.utils import atomic_json, fingerprint, sha256


def input_dates(config):
    values = feather.read_table(config.path("data_dir") / "close.feather", columns=["date"])["date"].to_pylist()
    dates = [value.strftime("%Y-%m-%d") for value in values]
    return dates[dates.index(config.start) - 1:dates.index(config.end) + 1]


def refresh_report(config, directory):
    saved = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    config = Config.from_dict(saved["config"], config.root)
    sources = json.loads((directory / "sources.json").read_text(encoding="utf-8"))
    audit = json.loads((directory / "data_audit.json").read_text(encoding="utf-8"))
    diagnostics = {"coverage": pd.read_csv(directory / "coverage.csv"),
                   "ic": pd.read_csv(directory / "rank_ic.csv"),
                   "ic_summary": pd.read_csv(directory / "ic_summary.csv"),
                   "groups": pd.read_csv(directory / "group_forward_returns.csv"),
                   "group_curves": pd.read_parquet(directory / "group_equity.parquet"),
                   "benchmark": pd.read_csv(directory / "benchmark_equity.csv"),
                   "correlation": pd.read_csv(directory / "factor_correlation.csv", index_col=0).to_numpy()}
    summaries, equities = [], {}
    by_factor = {item["factor"]: item for item in saved["ranking"]}
    for factor in FACTORS:
        run_dir = config.path("output_dir") / by_factor[factor.id]["run_id"]
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        equity, trades = (pd.read_parquet(run_dir / f"{name}.parquet") for name in ("equity", "trades"))
        summaries.append(enrich_result(summary, equity, trades, factor, diagnostics, run_dir))
        equities[factor.id] = equity
    report_fingerprint = fingerprint({"config": config.public(), "source": sources["fingerprint"],
                                     "runs": sorted(item["run_id"] for item in summaries),
                                     "analysis_code": sha256(config.root / "src/ashare_bt/factor_analytics.py"),
                                     "runner": sha256(Path(__file__))})
    write_study_report(directory, config, summaries, equities, diagnostics, audit, sources, report_fingerprint)


def verify_run(directory, config, data, features, status, replay=False):
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    equity = pd.read_parquet(directory / "equity.parquet")
    positions = pd.read_parquet(directory / "positions.parquet")
    trades = pd.read_parquet(directory / "trades.parquet")
    selection = pq.read_table(directory / "selection.parquet", columns=["date", "selected", "target_weight"]).to_pandas()
    assert len(equity) == len(data.trading_indices(config))
    assert equity.date.iloc[0] == config.start and equity.date.iloc[-1] == config.end
    assert (equity.positions <= config.max_positions).all()
    assert (equity.nav.astype(float) > 0).all()
    assert selection.target_weight.max() <= float(config.target_exposure) / config.max_positions + 1e-12
    assert selection.groupby("date").target_weight.sum().max() <= float(config.target_exposure) + 1e-10
    assert not summary["exit_planning"]["uses_future_market_data"]
    if len(trades):
        assert (pd.to_datetime(trades.date) > pd.to_datetime(trades.decision_date)).all()
        for (_, _), frame in trades.groupby(["date", "security"]):
            assert frame.notional.astype(float).sum() <= frame.capacity.astype(float).min() + .02
    record = {"run_id": summary["run_id"], "trading_days": len(equity), "positions_within_limit": True,
              "target_weights_within_limit": True, "capacity_within_limit": True,
              "daily_information_exit": True, "ending_positions": int(equity.positions.iloc[-1])}
    if replay:
        replay_features = FeatureBundle({key: value for key, value in features.values.items() if not key.startswith("classic_")},
                                        features.flags, features.metadata, features.profile)
        result = run(config, data=data, features=replay_features, status=status, output=False)
        for name in ("orders", "trades", "journal", "equity", "positions", "fees", "exits"):
            saved = pd.read_parquet(directory / f"{name}.parquet")
            pd.testing.assert_frame_equal(saved, getattr(result, name), check_exact=True)
        record["replay_tables_identical"] = ["orders", "trades", "journal", "equity", "positions", "fees", "exits"]
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs/classic_factors.toml"))
    parser.add_argument("--prepare-only", action="store_true", help="准备和校验原始响应缓存")
    parser.add_argument("--report-only", action="store_true", help="从已保存的净值和诊断重建报告与图表")
    parser.add_argument("--verify-replay", action="store_true", help="完整重放并逐表核对十项策略")
    args = parser.parse_args(argv)
    config = Config.load(args.config)
    directory = config.root / "reports" / "classic-factors"
    directory.mkdir(parents=True, exist_ok=True)
    if args.report_only:
        if args.prepare_only or args.verify_replay:
            raise ValueError("报告重建请单独使用 --report-only")
        refresh_report(config, directory)
        print(str(directory / "report.md"), flush=True)
        return 0
    progress_path = config.root / "work" / "classic-factors-progress.json"
    def progress(**record):
        atomic_json(progress_path, record)
        if record.get("stage") != "backtest" or record.get("done", 0) % 50 == 1 or record.get("done") == record.get("total"):
            print(json.dumps(record, ensure_ascii=False), flush=True)
    sources = fetch_sources(config, input_dates(config), progress)
    if sources["errors"]:
        atomic_json(directory / "source_gaps.json", sources["errors"])
        raise RuntimeError(f"原始数据准备尚有 {len(sources['errors'])} 项待重试；缓存已保存，可运行同一命令续传")
    if args.prepare_only:
        print(json.dumps({"prepared_queries": len(sources["manifests"]), "source_fingerprint": sources["fingerprint"]}), flush=True)
        return 0
    select_device(config.device)
    data = load_data(config, progress)
    features = compute_features(data, config, progress)
    status = StatusProvider(data.dates, data.securities, config.path("status_dir"))
    status_coverage = status.coverage(status_requirement(data, features, config))
    if status_coverage["unknown_stock_days"]:
        raise RuntimeError("历史 ST 状态需要补齐：" + str(status_coverage["unknown_stock_days"]))
    fundamentals, audit = load_fundamentals(data, sources, config.root / "data" / "classic_factors")
    features, scores, common = compute_classic_features(data, features, fundamentals, config, sources["fingerprint"], progress)
    classic = features.metadata["classic_factors"]
    parameters = {**config.parameters, "factor_cache": str(Path(classic["cache"]).relative_to(config.root)),
                  "factor_cache_sha256": classic["cache_sha256"], "factor_fingerprint": classic["fingerprint"]}
    config = replace(config, parameters=parameters)
    source_dir = config.root / "src" / "ashare_bt"
    execution_fingerprint = fingerprint({"config": config.public(), "market": data.fingerprint,
                                        "status": status.fingerprint, "factors": classic["fingerprint"],
                                        "code": {name: sha256(source_dir / name) for name in
                                                 ("engine.py", "execution.py", "ledger.py", "rules.py", "classic_factors.py", "strategies.py")}})
    study_fingerprint = fingerprint({"execution": execution_fingerprint,
                                     "analysis_code": sha256(source_dir / "factor_analytics.py"),
                                     "runner": sha256(Path(__file__))})
    atomic_json(directory / "config.json", config.public())
    atomic_json(directory / "sources.json", sources)
    atomic_json(directory / "data_audit.json", audit)
    diagnostics = factor_diagnostics(data, features, status, config, scores, common, fundamentals, directory, progress)
    summaries, equities, validation = [], {}, []
    for factor in FACTORS:
        factor_config = replace(config, strategy="classic_" + factor.id, name=f"经典{factor.name}因子 · 10亿100只")
        run_id = "classic-" + factor.id.replace("_", "-") + "-" + execution_fingerprint[:10]
        destination = factor_config.path("output_dir") / run_id
        if (destination / "summary.json").exists():
            saved_config = json.loads((destination / "config.json").read_text(encoding="utf-8"))
            if saved_config != factor_config.public():
                raise ValueError("已有运行配置需要核验：" + run_id)
            summary = json.loads((destination / "summary.json").read_text(encoding="utf-8"))
            equity = pd.read_parquet(destination / "equity.parquet")
            trades = pd.read_parquet(destination / "trades.parquet")
        else:
            if destination.exists():
                raise RuntimeError("检测到待完成的运行目录，请保留检查并移动后续传：" + str(destination))
            progress(stage="strategy", done=len(summaries), total=10, message="回测" + factor.name)
            strategy = ClassicFactorStrategy(factor_config, scores, data.dates, common)
            result = run(factor_config, strategy, data=data, features=features, status=status, run_id=run_id, progress=progress)
            summary, equity, trades = result.summary, result.equity, result.trades
        summary = enrich_result(summary, equity, trades, factor, diagnostics, destination)
        summaries.append(summary)
        equities[factor.id] = equity
        validation.append(verify_run(destination, factor_config, data, features, status, args.verify_replay))
        atomic_json(directory / "validation.json", clean({"completed": len(validation), "total": 10,
                    "status_coverage": status_coverage, "device": features.metadata,
                    "runs": validation, "source_requests": sources["requested"], "source_errors": sources["errors"]}))
        progress(stage="strategy_completed", done=len(summaries), total=10,
                 message=f"{factor.name}：净收益 {summary['return']:.2%}，最大回撤 {summary['max_drawdown']:.2%}，平均仓位 {summary['average_exposure']:.2%}")
    summaries = write_study_report(directory, config, summaries, equities, diagnostics, audit, sources, study_fingerprint)
    progress(stage="completed", done=10, total=10, message="十大经典因子评估与报告已完成")
    print(json.dumps({"report": str(directory / "report.md"), "ranking": [
        {key: item[key] for key in ("factor_name", "return", "max_drawdown", "sharpe", "average_exposure")} for item in summaries]}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise
