"""经典因子的时点基本面、MPS 特征与等权排名组合。"""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from decimal import Decimal
from itertools import groupby
from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch

from .compute import FeatureBundle, gpu_lock, rolling, select_device, synchronize
from .factor_sources import PublicDataCache
from .strategies import Strategy, StrategySpec, TargetWeights, stable_rank
from .utils import atomic_json, dec, fingerprint, sha256


@dataclass(frozen=True)
class ClassicFactor:
    id: str
    name: str
    formula: str
    direction: str


FACTORS = (
    ClassicFactor("value", "价值", "最新可用归母净资产 / 当日总市值", "高"),
    ClassicFactor("size", "小市值", "当日总市值", "低"),
    ClassicFactor("momentum", "中期动量", "复权收盘价[t-21] / 复权收盘价[t-252] - 1", "高"),
    ClassicFactor("reversal", "短期反转", "最近 5 日复权收益", "低"),
    ClassicFactor("low_volatility", "低波动", "最近 60 日日收益总体标准差", "低"),
    ClassicFactor("quality", "质量", "归母净利润 TTM / 同期及上年同期平均归母净资产", "高"),
    ClassicFactor("growth", "成长", "营业收入 TTM / 上年同期营业收入 TTM - 1", "高"),
    ClassicFactor("dividend", "红利", "近 365 日已实施税前现金分红总额 / 当日总市值", "高"),
    ClassicFactor("liquidity", "流动性溢价", "最近 20 日 mean(abs(日收益) / 成交额)", "高"),
    ClassicFactor("investment", "保守投资", "总资产 / 上年同季总资产 - 1", "低"),
)
FACTOR_MAP = {factor.id: factor for factor in FACTORS}


def number(value):
    try:
        value = float(value)
        return value if np.isfinite(value) else np.nan
    except (ValueError, TypeError):
        return np.nan


def day(value):
    try:
        stamp = pd.Timestamp(value)
        return None if pd.isna(stamp) else stamp.strftime("%Y-%m-%d")
    except (ValueError, TypeError):
        return None


def financial_record(raw, kind, source_key=""):
    """将当前版本保守推迟到公告与更新日期中较晚一日结束之后。"""
    period, notice, update = (day(raw.get(key)) for key in ("REPORT_DATE", "NOTICE_DATE", "UPDATE_DATE"))
    if period is None or notice is None or update is None or raw.get("CURRENCY") != "CNY":
        return None
    return {"security": raw.get("SECUCODE"), "kind": kind, "report_period": period,
            "notice_date": notice, "update_date": update, "known_day": max(period, notice, update),
            "parent_equity": number(raw.get("TOTAL_PARENT_EQUITY")), "assets": number(raw.get("TOTAL_ASSETS")),
            "revenue": number(raw.get("TOTAL_OPERATE_INCOME")), "profit": number(raw.get("PARENT_NETPROFIT")),
            "source_key": source_key}


def previous_year(period):
    return str(int(period[:4]) - 1) + period[4:]


def ttm(records, period, field):
    current = records.get(period, {}).get(field, np.nan)
    if period.endswith("12-31"):
        return current
    last_annual = str(int(period[:4]) - 1) + "-12-31"
    return current + records.get(last_annual, {}).get(field, np.nan) - records.get(previous_year(period), {}).get(field, np.nan)


def financial_state(balance, income):
    result = {key: np.nan for key in ("book", "quality", "growth", "investment", "balance_period", "income_period")}
    if balance:
        period = max(balance)
        current = balance[period]
        result["balance_period"] = period
        book = current.get("parent_equity", np.nan)
        result["book"] = book if book > 0 else np.nan
        old_assets = balance.get(previous_year(period), {}).get("assets", np.nan)
        assets = current.get("assets", np.nan)
        if old_assets > 0 and assets >= 0:
            result["investment"] = assets / old_assets - 1
    if income:
        period = max(income)
        result["income_period"] = period
        revenue, old_revenue = ttm(income, period, "revenue"), ttm(income, previous_year(period), "revenue")
        if old_revenue > 0 and revenue >= 0:
            result["growth"] = revenue / old_revenue - 1
        equity = balance.get(period, {}).get("parent_equity", np.nan)
        old_equity = balance.get(previous_year(period), {}).get("parent_equity", np.nan)
        if equity > 0 and old_equity > 0:
            result["quality"] = ttm(income, period, "profit") / ((equity + old_equity) / 2)
    return result


def financial_panels(records, dates, securities):
    shape = len(dates), len(securities)
    arrays = {name: np.full(shape, np.nan, dtype=np.float32) for name in
              ("book", "quality", "growth", "investment", "financial_age_days")}
    index = {security: column for column, security in enumerate(securities)}
    grouped = {}
    for record in records:
        if record["security"] in index:
            grouped.setdefault(record["security"], []).append(record)
    dates64 = np.asarray(dates, dtype="datetime64[D]")
    for security, items in grouped.items():
        column = index[security]
        items.sort(key=lambda record: (record["known_day"], record["kind"], record["report_period"], record["source_key"]))
        balance, income = {}, {}
        groups = [(known, list(batch)) for known, batch in groupby(items, key=lambda record: record["known_day"])]
        for event_index, (known, batch) in enumerate(groups):
            for record in batch:
                destination = balance if record["kind"] == "balance" else income
                destination[record["report_period"]] = record
            start = bisect_right(dates, known)
            stop = bisect_right(dates, groups[event_index + 1][0]) if event_index + 1 < len(groups) else len(dates)
            if start >= stop:
                continue
            state = financial_state(balance, income)
            for name in ("book", "quality", "growth", "investment"):
                arrays[name][start:stop, column] = state[name]
            periods = [state[name] for name in ("balance_period", "income_period") if isinstance(state[name], str)]
            if periods:
                arrays["financial_age_days"][start:stop, column] = (dates64[start:stop] - np.datetime64(min(periods))).astype(int)
    return arrays


def dividend_panel(records, dates, securities, complete=True):
    values = np.zeros((len(dates), len(securities)), dtype=np.float64)
    if not complete:
        values[:] = np.nan
        return values.astype(np.float32), {"complete": False, "events": 0, "invalid_events": 0}
    index = {security: column for column, security in enumerate(securities)}
    selected = {}
    invalid = 0
    for raw in records:
        security = raw.get("SECUCODE")
        ex_date, notice = day(raw.get("EX_DIVIDEND_DATE")), day(raw.get("NOTICE_DATE"))
        if security not in index or ex_date is None or "实施" not in str(raw.get("ASSIGN_PROGRESS", "")):
            continue
        key = security, day(raw.get("REPORT_DATE")), ex_date
        previous = selected.get(key)
        if previous is None or str(raw.get("NOTICE_DATE")) > str(previous.get("NOTICE_DATE")):
            selected[key] = raw
    for (security, _, ex_date), raw in selected.items():
        notice = day(raw.get("NOTICE_DATE"))
        update = day(raw.get("UPDATE_DATE"))
        known = max([value for value in (notice, update) if value], default=None)
        if known is None:
            invalid += 1
            values[:, index[security]] = np.nan
            continue
        start = max(bisect_left(dates, ex_date), bisect_right(dates, known))
        stop = bisect_left(dates, str((pd.Timestamp(ex_date) + pd.Timedelta(days=365)).date()))
        amount = number(raw.get("PRETAX_BONUS_RMB")) / 10 * number(raw.get("TOTAL_SHARES"))
        if not np.isfinite(amount) or amount < 0:
            invalid += 1
            values[start:stop, index[security]] = np.nan
        else:
            values[start:stop, index[security]] += amount
    return values.astype(np.float32), {"complete": True, "events": len(selected), "invalid_events": invalid,
                                      "cash_unit": "PRETAX_BONUS_RMB 为每 10 股元，TOTAL_SHARES 为股"}


def load_fundamentals(data, source_result, directory):
    shares = np.full(data.shape, np.nan, dtype=np.float32)
    financial, dividends, alignments = [], [], []
    rejected_financial = 0
    for manifest in source_result["manifests"]:
        kind = manifest["query"]["kind"]
        records = PublicDataCache.read(manifest)
        if kind == "shares":
            for raw in records:
                row, column = data.date_index.get(day(raw.get("TRADE_DATE"))), data.index.get(raw.get("SECUCODE"))
                if row is None or column is None:
                    continue
                value = number(raw.get("TOTAL_SHARES"))
                if value > 0:
                    shares[row, column] = value
                close = number(raw.get("CLOSE_PRICE"))
                local = data.arrays["close"][row, column]
                if np.isfinite(close) and close > 0 and np.isfinite(local):
                    alignments.append(abs(local / close - 1))
        elif kind in {"balance", "income"}:
            for raw in records:
                if raw.get("SECUCODE") not in data.index:
                    continue
                record = financial_record(raw, kind, manifest["key"])
                if record:
                    financial.append(record)
                else:
                    rejected_financial += 1
        elif kind == "dividend":
            dividends.extend(records)
    panels = financial_panels(financial, data.dates, data.securities)
    complete = any(m["query"]["kind"] == "dividend" for m in source_result["manifests"]) and all(
        error["query"]["kind"] != "dividend" for error in source_result["errors"])
    panels["dividend_cash"], dividend_audit = dividend_panel(dividends, data.dates, data.securities, complete)
    panels["shares"] = shares
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    normalized = pd.DataFrame(financial)
    normalized.to_parquet(directory / "financial_records.parquet", index=False)
    pd.DataFrame(dividends).to_parquet(directory / "dividend_records.parquet", index=False)
    audit = {"financial_records": len(financial), "rejected_financial_records": rejected_financial,
             "financial_records_updated_after_notice": sum(r["update_date"] > r["notice_date"] for r in financial),
             "financial_records_known_after_end": sum(r["known_day"] >= data.dates[-1] for r in financial),
             "dividends": dividend_audit, "source_errors": source_result["errors"],
             "price_alignment": {"observations": len(alignments), "differences_over_1_percent": sum(v > .01 for v in alignments),
                                 "median_relative_difference": float(np.median(alignments)) if alignments else None},
             "version_policy": "max(报告期, 公告日期, 更新日期) 日末之后可用；按报告期保留当时最新版本",
             "standard": "保守的探索性评估；源更新时间近似历史版本可用时间"}
    atomic_json(directory / "data_audit.json", audit)
    return panels, audit


def compute_classic_features(data, features, fundamentals, config, source_fingerprint, progress=lambda **kw: None):
    device = select_device(config.device)
    key = fingerprint({"schema": 1, "market": data.fingerprint, "source": source_fingerprint,
                       "code": sha256(Path(__file__)), "block_days": config.block_days, "device": str(device)})
    cache = config.root / "data" / "classic_factors" / "features" / f"{key}.npz"
    arrays = {factor.id: np.full(data.shape, np.nan, dtype=np.float32) for factor in FACTORS}
    if cache.exists():
        with np.load(cache, allow_pickle=False) as saved:
            arrays = {factor.id: saved[factor.id] for factor in FACTORS}
    else:
        with gpu_lock(config, progress, None):
            for start in range(0, data.shape[0], config.block_days):
                stop = min(start + config.block_days, data.shape[0])
                lower = max(0, start - 253)
                offset = start - lower
                tensor = lambda values: torch.from_numpy(np.array(values[lower:stop], dtype=np.float32, order="C", copy=True)).to(device)
                raw_close = tensor(data.arrays["close"])
                adjusted = raw_close * tensor(data.arrays["cumadj"])
                amount = tensor(data.arrays["amt"])
                market_cap = raw_close * tensor(fundamentals["shares"])
                good_cap = torch.isfinite(market_cap) & (market_cap > 0)
                ratio = lambda numerator: torch.where(good_cap, numerator / market_cap, torch.nan)
                ret = torch.full_like(adjusted, torch.nan)
                ret[1:] = adjusted[1:] / adjusted[:-1] - 1
                momentum = torch.full_like(adjusted, torch.nan)
                momentum[252:] = adjusted[231:-21] / adjusted[:-252] - 1
                reversal = torch.full_like(adjusted, torch.nan)
                reversal[5:] = adjusted[5:] / adjusted[:-5] - 1
                outputs = {"value": ratio(tensor(fundamentals["book"])),
                           "size": torch.where(good_cap, -market_cap, torch.nan), "momentum": momentum,
                           "reversal": -reversal, "low_volatility": -rolling(ret, 60, "std"),
                           "quality": tensor(fundamentals["quality"]), "growth": tensor(fundamentals["growth"]),
                           "dividend": ratio(tensor(fundamentals["dividend_cash"])),
                           "liquidity": rolling(torch.where(amount > 0, ret.abs() / amount, torch.nan), 20),
                           "investment": -tensor(fundamentals["investment"])}
                for name, values in outputs.items():
                    arrays[name][start:stop] = values[offset:].detach().cpu().numpy()
                synchronize(device)
                progress(stage="classic_factors", done=stop, total=data.shape[0], message=f"MPS 十因子 {stop}/{data.shape[0]}")
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, **arrays, dates=np.asarray(data.dates), securities=np.asarray(data.securities),
                            fingerprint=np.asarray(key), market_fingerprint=np.asarray(data.fingerprint))
    common = np.logical_and.reduce([np.isfinite(arrays[factor.id]) for factor in FACTORS])
    values = {**features.values, **{"classic_" + name: array for name, array in arrays.items()},
              "classic_common": common.astype(np.float32)}
    metadata = {**features.metadata, "classic_factors": {"fingerprint": key, "source_fingerprint": source_fingerprint,
                                                        "cache": str(cache), "cache_sha256": sha256(cache), "device": str(device)}}
    bundle = FeatureBundle(values, features.flags, metadata, features.profile)
    return bundle, arrays, common


class ClassicFactorStrategy(Strategy):
    def __init__(self, config, panels=None, dates=None, common=None):
        self.config = config
        self.factor = FACTOR_MAP[config.strategy.removeprefix("classic_")]
        self.spec = StrategySpec(config.strategy, self.factor.name, "经典因子", self.factor.formula,
                                 "按共同股票池中的指标排名选取前列证券。", "样本、时点数据和交易容量共同影响表现。")
        self.panels, self.common = panels, common
        self.date_index = {value: index for index, value in enumerate(dates)} if dates is not None else None
        self.calls = 0
        self.last_scores = None
        self.last_score_date = None

    def _load(self):
        path = self.config.root / self.config.parameters["factor_cache"]
        if sha256(path) != self.config.parameters["factor_cache_sha256"]:
            raise ValueError("因子快照哈希需要核验")
        with np.load(path, allow_pickle=False) as saved:
            if str(saved["fingerprint"]) != self.config.parameters["factor_fingerprint"]:
                raise ValueError("因子快照版本需要核验")
            self.panels = {factor.id: saved[factor.id] for factor in FACTORS}
            self.common = np.logical_and.reduce([np.isfinite(self.panels[factor.id]) for factor in FACTORS])
            self.date_index = {value: index for index, value in enumerate(saved["dates"].tolist())}

    def score(self, history):
        try:
            score, common = history.factor("classic_" + self.factor.id), history.factor("classic_common") == 1
        except KeyError:
            if self.panels is None:
                self._load()
            row = self.date_index[history.date]
            score, common = self.panels[self.factor.id][row], self.common[row]
        return np.where(common, score, np.nan).astype(float)

    def on_close(self, history, portfolio):
        due = self.calls % self.config.rebalance_days == 0
        self.calls += 1
        if not due:
            return TargetWeights({}, "等待五日调仓", False)
        scores = self.score(history)
        indices = stable_rank(scores, history.eligible, history.securities)[:self.config.max_positions]
        weight = (dec(self.config.target_exposure) / self.config.max_positions).quantize(Decimal("0.000000000001"))
        self.last_scores, self.last_score_date = scores.copy(), history.date
        return TargetWeights({history.securities[index]: weight for index in indices}, self.factor.name + "五日排名调仓")
