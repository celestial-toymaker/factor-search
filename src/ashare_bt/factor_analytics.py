"""因子 IC、理论分组组合、容量解释及中文比较报告。"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json
import re
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import torch

from .classic_factors import FACTORS
from .compute import select_device
from .engine import eligibility
from .factor_sources import ENDPOINT
from .rules import RULE_SOURCES
from .strategies import stable_rank
from .utils import atomic_json, dec, sha256


def clean(value):
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def curve_metrics(nav):
    values = np.r_[1., np.asarray(nav, dtype=float)]
    changes = values[1:] / values[:-1] - 1
    drawdown = values / np.maximum.accumulate(values) - 1
    volatility = changes.std(ddof=1) if len(changes) > 1 else 0.
    annual = values[-1] ** (252 / len(changes)) - 1 if len(changes) and values[-1] > 0 else np.nan
    maximum = -drawdown.min()
    return {"return": values[-1] - 1, "annual_return": annual, "max_drawdown": maximum,
            "annual_volatility": volatility * np.sqrt(252),
            "sharpe": changes.mean() / volatility * np.sqrt(252) if volatility > 0 else 0.,
            "calmar": annual / maximum if maximum > 0 else np.nan,
            "positive_daily_fraction": float((changes > 0).mean()) if len(changes) else np.nan}


def rank_correlations(matrix, device):
    """平均名次处理同分，MPS 计算相关矩阵。"""
    ranks = pd.DataFrame(np.asarray(matrix).T).rank(method="average").to_numpy(dtype=np.float32)
    values = torch.from_numpy(np.ascontiguousarray(ranks)).to(device)
    values = values - values.mean(dim=0)
    norms = torch.sqrt((values * values).sum(dim=0))
    denominator = norms[:, None] * norms[None, :]
    product = values.T @ values
    return torch.where(denominator > 0, product / denominator, torch.nan).cpu().numpy()


def theoretical_curve(data, config, selections):
    """95% 等权理论组合，开盘重配；报价中断时延续持仓和有效估值。"""
    indices = data.trading_indices(config)
    adjusted_open = data.arrays["open"] * data.arrays["cumadj"]
    adjusted_close = data.arrays["close"] * data.arrays["cumadj"]
    units = np.zeros(data.shape[1], dtype=float)
    cash = 1.
    marks = np.zeros(data.shape[1], dtype=float)
    first = indices[0] - 1
    for row in range(first + 1):
        valid = np.isfinite(adjusted_close[row]) & (adjusted_close[row] > 0)
        marks[valid] = adjusted_close[row, valid]
    rows = []
    for row in indices:
        opening = adjusted_open[row]
        tradable = np.isfinite(opening) & (opening > 0)
        open_marks = np.where(tradable, opening, marks)
        nav_open = cash + np.dot(units, open_marks)
        if row - 1 in selections:
            selected = np.asarray(selections[row - 1], dtype=int)
            cash += np.dot(units[tradable], open_marks[tradable])
            units[tradable] = 0
            if len(selected):
                allowed = selected[tradable[selected]]
                per_stock = nav_open * float(config.target_exposure) / len(selected)
                desired = np.full(len(allowed), per_stock)
                if desired.sum() > cash and desired.sum() > 0:
                    desired *= max(0., cash) / desired.sum()
                units[allowed] = desired / open_marks[allowed]
                cash -= desired.sum()
        valid_close = np.isfinite(adjusted_close[row]) & (adjusted_close[row] > 0)
        marks = np.where(valid_close, adjusted_close[row], open_marks)
        market_value = np.dot(units, marks)
        nav = cash + market_value
        rows.append({"date": data.dates[row], "unit_nav": nav, "cash": cash,
                     "exposure": market_value / nav if nav > 0 else 0})
    return pd.DataFrame(rows)


def factor_diagnostics(data, features, status, config, scores, common, fundamentals, directory, progress=lambda **kw: None):
    device = select_device(config.device)
    indices = data.trading_indices(config)
    first, last = indices[0] - 1, indices[-1]
    decision_rows = set(range(first, last, config.rebalance_days))
    base_masks = {}
    coverage, ic_rows, group_rows = [], [], []
    selections = {"benchmark": {}}
    for factor in FACTORS:
        for group in range(1, 6):
            selections[f"{factor.id}_Q{group}"] = {}
    correlations, correlation_counts = np.zeros((10, 10)), np.zeros((10, 10), dtype=int)
    adjusted_open = data.arrays["open"] * data.arrays["cumadj"]
    securities = np.asarray(data.securities)
    for number, row in enumerate(range(first, last + 1), 1):
        base, _ = eligibility(data, features, status, config, row)
        base_masks[row] = base
        eligible = base & common[row]
        coverage_row = {"date": data.dates[row], "base_eligible": int(base.sum()), "common_eligible": int(eligible.sum())}
        for factor in FACTORS:
            coverage_row[factor.id] = int((base & np.isfinite(scores[factor.id][row])).sum())
        ages = fundamentals["financial_age_days"][row, eligible]
        coverage_row["median_financial_age_days"] = float(np.median(ages)) if len(ages) else np.nan
        coverage.append(coverage_row)
        ranked_groups = {}
        if row in decision_rows:
            selections["benchmark"][row] = np.flatnonzero(eligible)
            for factor in FACTORS:
                ranked = stable_rank(scores[factor.id][row], eligible, securities)
                ranked_groups[factor.id] = {5 - group: values for group, values in enumerate(np.array_split(ranked, 5))}
                for group, values in ranked_groups[factor.id].items():
                    selections[f"{factor.id}_Q{group}"][row] = values
            if eligible.sum() >= 5:
                matrix = np.stack([scores[factor.id][row, eligible] for factor in FACTORS])
                correlation = rank_correlations(matrix, device)
                valid = np.isfinite(correlation)
                correlations[valid] += correlation[valid]
                correlation_counts[valid] += 1
        if row + 6 <= last:
            with np.errstate(invalid="ignore", divide="ignore"):
                future = adjusted_open[row + 6] / adjusted_open[row + 1] - 1
            valid = eligible & np.isfinite(future) & (adjusted_open[row + 1] > 0) & (adjusted_open[row + 6] > 0)
            if valid.sum() >= 5:
                matrix = np.stack([*(scores[factor.id][row, valid] for factor in FACTORS), future[valid]])
                correlation = rank_correlations(matrix, device)
                for index, factor in enumerate(FACTORS):
                    ic_rows.append({"date": data.dates[row], "factor": factor.id, "rank_ic": float(correlation[index, -1]),
                                    "signal_count": int(eligible.sum()), "label_count": int(valid.sum()),
                                    "rebalance": row in decision_rows})
            if row in decision_rows:
                for factor in FACTORS:
                    for group, selected in ranked_groups[factor.id].items():
                        observed = selected[np.isfinite(future[selected]) & (adjusted_open[row + 1, selected] > 0)
                                            & (adjusted_open[row + 6, selected] > 0)]
                        group_rows.append({"date": data.dates[row], "factor": factor.id, "group": group,
                                           "signal_count": len(selected), "label_count": len(observed),
                                           "forward_return": float(future[observed].mean()) if len(observed) else np.nan})
        if number % 50 == 0 or row == last:
            progress(stage="diagnostics", done=number, total=last - first + 1, message=f"因子有效性与覆盖 {number}/{last-first+1}")
    coverage = pd.DataFrame(coverage)
    ic = pd.DataFrame(ic_rows, columns=["date", "factor", "rank_ic", "signal_count", "label_count", "rebalance"])
    grouped = pd.DataFrame(group_rows, columns=["date", "factor", "group", "signal_count", "label_count", "forward_return"])
    correlation = np.divide(correlations, correlation_counts, out=np.full_like(correlations, np.nan), where=correlation_counts > 0)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    coverage.to_csv(directory / "coverage.csv", index=False, encoding="utf-8-sig")
    ic.to_csv(directory / "rank_ic.csv", index=False, encoding="utf-8-sig")
    grouped.to_csv(directory / "group_forward_returns.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(correlation, index=[f.id for f in FACTORS], columns=[f.id for f in FACTORS]).to_csv(directory / "factor_correlation.csv", encoding="utf-8-sig")
    benchmark = theoretical_curve(data, config, selections["benchmark"])
    benchmark.to_csv(directory / "benchmark_equity.csv", index=False, encoding="utf-8-sig")
    group_curves = []
    ic_summary = []
    for factor in FACTORS:
        for group in range(1, 6):
            curve = theoretical_curve(data, config, selections[f"{factor.id}_Q{group}"])
            curve["factor"], curve["group"] = factor.id, group
            group_curves.append(curve)
        values = ic.loc[ic.factor == factor.id, "rank_ic"].dropna()
        rebalance = ic.loc[(ic.factor == factor.id) & ic.rebalance, "rank_ic"].dropna()
        std = values.std(ddof=1)
        ic_summary.append({"factor": factor.id, "rank_ic_mean": values.mean(), "rank_ic_std": std,
                           "icir": values.mean() / std if std > 0 else np.nan,
                           "positive_ic_fraction": (values > 0).mean() if len(values) else np.nan,
                           "ic_days": len(values), "rebalance_ic_mean": rebalance.mean(), "rebalance_ic_days": len(rebalance)})
        progress(stage="group_curves", done=len(ic_summary), total=10, message=f"理论五分组 {factor.name}")
    group_curves = pd.concat(group_curves, ignore_index=True)
    group_curves.to_parquet(directory / "group_equity.parquet", index=False)
    pd.DataFrame(ic_summary).to_csv(directory / "ic_summary.csv", index=False, encoding="utf-8-sig")
    return {"coverage": coverage, "ic": ic, "ic_summary": pd.DataFrame(ic_summary), "groups": grouped,
            "group_curves": group_curves, "benchmark": benchmark, "correlation": correlation}


def actual_exit_summary(summary, exits, trades, orders, positions):
    """按最后退出计划和真实成交汇总，逐日累计快照保留单次终态。"""
    end = summary["end"]
    active = exits.loc[exits.exit_active].sort_values(["date", "security"], kind="stable") if len(exits) else pd.DataFrame()
    plans = active.drop_duplicates("security", keep="last") if len(active) else active
    sells = trades.loc[(trades.side == "sell") & (trades.reason == "按规则计划退出")] if len(trades) else pd.DataFrame()
    exit_orders = orders.loc[(orders.side == "sell") & (orders.reason == "按规则计划退出")] if len(orders) else pd.DataFrame()
    final_positions = positions.loc[positions.date == end].set_index("security") if len(positions) else pd.DataFrame()
    rows = []
    for plan in plans.to_dict("records"):
        security, start = plan["security"], plan["planned_start"]
        executed = sells.loc[(sells.security == security) & (sells.date >= start) & (sells.date <= end)] if len(sells) and isinstance(start, str) else pd.DataFrame()
        submitted = exit_orders.loc[(exit_orders.security == security) & (exit_orders.execution_date <= end)] if len(exit_orders) else pd.DataFrame()
        remaining = dec(final_positions.loc[security, "quantity"]) if security in final_positions.index else dec(0)
        market_value = dec(final_positions.loc[security, "market_value"]) if security in final_positions.index else dec(0)
        gross = sum((dec(value) for value in executed.notional), dec(0)) if len(executed) else dec(0)
        fees = sum((dec(value) for value in executed.fees), dec(0)) if len(executed) else dec(0)
        reasons = []
        if remaining > 0 and len(submitted):
            latest = submitted.loc[submitted.execution_date == submitted.execution_date.max()]
            reasons = sorted(set(latest.loc[latest.remaining.map(dec) > 0, "last_reason"]) - {"", "开盘成交"})
        rows.append({"security": security, "triggered_on": plan["triggered_on"], "planned_start": start,
                     "actual_start": str(executed.date.min()) if len(executed) else None,
                     "last_sell_date": str(executed.date.max()) if len(executed) else None,
                     "sell_trade_count": len(executed), "gross_proceeds": str(gross), "sale_fees": str(fees),
                     "net_proceeds": str(gross - fees), "remaining_quantity": str(remaining),
                     "remaining_market_value": str(market_value),
                     "status": "已清仓" if remaining == 0 else "持仓保留",
                     "remaining_reason": "；".join(reasons)})
    columns = ["security", "triggered_on", "planned_start", "actual_start", "last_sell_date", "sell_trade_count",
               "gross_proceeds", "sale_fees", "net_proceeds", "remaining_quantity", "remaining_market_value", "status", "remaining_reason"]
    records = pd.DataFrame(rows, columns=columns)
    terminal = [record for record in rows if record["planned_start"] == end]
    earlier = [record for record in rows if isinstance(record["planned_start"], str) and record["planned_start"] < end]
    terminal_trades = sells.loc[(sells.date == end) & sells.security.isin([record["security"] for record in terminal])] if len(sells) else pd.DataFrame()
    reasons = {}
    for record in rows:
        if dec(record["remaining_quantity"]) > 0:
            for reason in filter(None, record["remaining_reason"].split("；")):
                reasons.setdefault(reason, set()).add(record["security"])
    gross = sum((dec(value) for value in terminal_trades.notional), dec(0)) if len(terminal_trades) else dec(0)
    costs = sum((dec(value) for value in terminal_trades.fees), dec(0)) if len(terminal_trades) else dec(0)
    result = {"terminal_trigger_dates": sorted({record["triggered_on"] for record in terminal if isinstance(record["triggered_on"], str)}),
              "terminal_actual_sell_date": str(terminal_trades.date.min()) if len(terminal_trades) else None,
              "terminal_planned_securities": len(terminal),
              "terminal_sold_securities": int(terminal_trades.security.nunique()) if len(terminal_trades) else 0,
              "terminal_gross_proceeds": str(gross), "terminal_sale_fees": str(costs), "terminal_net_proceeds": str(gross - costs),
              "terminal_cleared_securities": sum(dec(record["remaining_quantity"]) == 0 for record in terminal),
              "remaining_securities": sum(dec(record["remaining_quantity"]) > 0 for record in rows),
              "remaining_market_value": str(sum((dec(record["remaining_market_value"]) for record in rows), dec(0))),
              "earlier_planned_securities": len(earlier),
              "earlier_cleared_securities": sum(dec(record["remaining_quantity"]) == 0 for record in earlier),
              "remaining_reasons": [{"reason": reason, "securities": len(securities)} for reason, securities in sorted(reasons.items())],
              "basis": "按规则计划退出的真实成交；逐日快照按证券保留最后退出状态"}
    return result, records


def actual_exit_section(summary, actual):
    dates = "、".join(actual["terminal_trigger_dates"]) or "待触发"
    sold_date = actual["terminal_actual_sell_date"] or "待实际成交"
    remaining_value = dec(actual["remaining_market_value"])
    share = remaining_value / dec(summary["ending_nav"]) if dec(summary["ending_nav"]) > 0 else dec(0)
    labels = []
    for record in actual["remaining_reasons"]:
        reason = record["reason"]
        if "容量" in reason:
            reason = "共享容量或合法数量限制"
        elif "当日成交额" in reason:
            reason = "当日成交额有效性限制"
        labels.append(f"{reason} {record['securities']} 只")
    reason_text = "；".join(labels) or ("持仓及末次委托详见退出摘要" if actual["remaining_securities"] else "持仓已结清")
    return ("## 2. 实际退出记录\n\n"
            f"**退出方式：{summary['exit_label']}。** {summary['exit_planning']['disclosure']}\n\n"
            f"期末退出于 **{dates}** 触发，计划退出 **{actual['terminal_planned_securities']} 只**。\n\n"
            "| 退出项目 | 实际记录 |\n|---|---:|\n"
            f"| 期末实际卖出 | {sold_date}，{actual['terminal_sold_securities']} 只，成交额 {dec(actual['terminal_gross_proceeds'])/10000:,.2f} 万元 |\n"
            f"| 期末完成清仓 | {actual['terminal_cleared_securities']} 只 |\n"
            f"| 期末保留持仓 | {actual['remaining_securities']} 只，市值 {remaining_value/10000:,.2f} 万元，占净资产 {share:.2%} |\n"
            f"| 主要剩余原因 | {reason_text} |\n\n"
            f"期内另有 {actual['earlier_planned_securities']} 只证券更早触发规则退出，其中 {actual['earlier_cleared_securities']} 只已清仓。"
            "以上按规则退出计划汇总实际成交。"
            "[逐证券实际退出摘要](exit_summary.csv)记录触发、实际卖出、费用及剩余持仓；"
            "[原始退出记录](exits.parquet)保留逐日过程。剩余持仓按期末有效行情计入净资产。\n\n")


def enrich_result(summary, equity, trades, factor, diagnostics, directory):
    metrics = curve_metrics(equity.unit_nav)
    prior = pd.Series(np.r_[float(summary["initial_cash"]), equity.nav.to_numpy(float)[:-1]], index=equity.date)
    if len(trades):
        notional = trades.assign(value=trades.notional.astype(float)).groupby("date").value.sum()
        turnover = float((notional / prior.reindex(notional.index)).sum()) * .5 * 252 / len(equity)
    else:
        turnover = 0.
    ic_record = diagnostics["ic_summary"].set_index("factor").loc[factor.id].to_dict()
    selected = diagnostics["group_curves"]
    q5 = selected[(selected.factor == factor.id) & (selected.group == 5)].unit_nav.iloc[-1]
    q1 = selected[(selected.factor == factor.id) & (selected.group == 1)].unit_nav.iloc[-1]
    extra = {**metrics, **ic_record, "factor": factor.id, "factor_name": factor.name,
             "annual_one_way_turnover": turnover, "q5_return": q5 - 1, "q1_return": q1 - 1,
             "q5_q1_return_difference": q5 - q1,
             "benchmark_return": float(diagnostics["benchmark"].unit_nav.iloc[-1] - 1),
             "excess_return_difference": metrics["return"] - float(diagnostics["benchmark"].unit_nav.iloc[-1] - 1),
             "performance_status": "有效成交" if summary["trade_count"] else "现金组合，待有效成交"}
    directory = Path(directory)
    actual, exit_records = actual_exit_summary(summary, pd.read_parquet(directory / "exits.parquet"), trades,
                                               pd.read_parquet(directory / "orders.parquet"), pd.read_parquet(directory / "positions.parquet"))
    exit_records.to_csv(directory / "exit_summary.csv", index=False, encoding="utf-8-sig")
    updated = clean({**summary, **extra, "factor_definition": asdict(factor), "actual_exit": actual})
    atomic_json(Path(directory) / "summary.json", updated)
    report_path = Path(directory) / "report.md"
    report = report_path.read_text(encoding="utf-8")
    start = report.index("## 2.")
    stop = report.index("## 3.")
    report = report[:start] + actual_exit_section(summary, actual) + report[stop:]
    start = report.index("## 6. 数据与研究假设")
    stop = report.index("## 7. 规则来源与复现")
    assumptions = "\n".join("- " + text for text in summary["research_assumptions"])
    section = ("## 6. 数据与研究假设\n\n"
               f"本策略使用{factor.name}因子，指标为「{factor.formula}」，优先选择指标{factor.direction}的证券。"
               "十因子共同有效股票池内按分数排序，同分时按证券代码升序。每 5 个交易日调仓，"
               "最多 100 只，每只目标权重 0.95%；名额不足时保留现金。\n\n"
               "行情和累计复权因子来自[课程数据 Release](https://github.com/aslan9/pku_quantllm/releases/tag/data-backtest-20261016)，"
               "历史 ST 状态来自 BaoStock 0.9.4 的 query_daily_history_k_AStock 缓存。"
               "财务、历史总股本和已实施分红来自东方财富数据接口。总市值按本地原始收盘价乘以当日总股本计算。"
               "财务因子使用结构化资产负债表与利润表字段，其中成长采用营业总收入 TOTAL_OPERATE_INCOME。\n\n"
               "财务记录以报告期维护各自版本，从 max(REPORT_DATE, NOTICE_DATE, UPDATE_DATE) 日期结束后的下一交易日开始使用。"
               "本研究采用当前抓取版本加保守时间推迟的历史版本近似；完整历史修订过程仍需历史时点数据支持。"
               "因此本结果采用保守探索性口径；财务信息滞后和共同样本筛选会影响收益。"
               "[十因子比较报告第 3 节](../../reports/classic-factors/report.md)列明全部数据源、接口字段、时间范围、抓取记录及溯源边界；"
               "[原始请求索引](../../reports/classic-factors/sources.json)保留每项请求的 URL、抓取时间和哈希。\n\n"
               f"{assumptions}\n\n"
               f"历史状态覆盖 {summary['status_coverage']['covered_stock_days']:,} / "
               f"{summary['status_coverage']['required_stock_days']:,} 个待选股票日。预热自 {summary['warmup']['first']} 开始，"
               f"首批决策日为 {summary['warmup']['last_decision']}。计算设备为 {summary['compute']['device']}，"
               "资金与费用采用 Decimal 记账。因子快照路径和哈希保存在运行配置中。\n\n")
    report = report[:start] + section + report[stop:]
    report = report.replace("策略与参考组合的累计收益曲线", "策略累计收益曲线")
    report = report.replace("保存逐日容量路径、最晚启动日、实际卖出进度和剩余数量", "保存逐日退出触发、实际卖出进度和剩余数量")
    report_path.write_text(report, encoding="utf-8")
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["report"].update({"document_sha256": sha256(report_path), "actual_exit_summary": "exit_summary.csv",
                               "actual_exit_summary_sha256": sha256(directory / "exit_summary.csv"),
                               "enrichment_generator_sha256": sha256(Path(__file__))})
    atomic_json(manifest_path, manifest)
    return updated


def period_returns(records):
    rows = []
    for factor, equity in records.items():
        frame = equity.copy()
        frame["period"] = pd.PeriodIndex(frame.date, freq="Q").astype(str)
        frame["year"] = pd.to_datetime(frame.date).dt.year.astype(str)
        for kind in ("year", "period"):
            for label, group in frame.groupby(kind, sort=True):
                start = group.index[0]
                previous = frame.unit_nav.iloc[start - 1] if start else 1.
                rows.append({"factor": factor, "kind": kind, "period": label,
                             "return": group.unit_nav.iloc[-1] / previous - 1})
    return pd.DataFrame(rows)


def charts(directory, summaries, equities, diagnostics):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.font_manager import FontProperties
    from matplotlib import colormaps
    directory = Path(directory)
    font_path = Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf")
    font = FontProperties(fname=str(font_path), size=9) if font_path.exists() else FontProperties(size=9)
    names = {factor.id: factor.name for factor in FACTORS}
    colors = colormaps["tab10"](np.arange(10))
    def save(figure, filename):
        FigureCanvasAgg(figure).print_png(directory / filename)
        figure.clear()
    figure = Figure(figsize=(13, 8), dpi=150)
    axes = figure.subplots(2, 1, sharex=True)
    for index, item in enumerate(summaries):
        equity = equities[item["factor"]]
        dates, nav = pd.to_datetime(equity.date), equity.unit_nav.to_numpy(float)
        axes[0].plot(dates, (nav - 1) * 100, label=item["factor_name"], color=colors[index], linewidth=1.2)
        peak = np.maximum.accumulate(np.r_[1., nav])[1:]
        axes[1].plot(dates, (nav / peak - 1) * 100, color=colors[index], linewidth=1)
    benchmark = diagnostics["benchmark"]
    axes[0].plot(pd.to_datetime(benchmark.date), (benchmark.unit_nav - 1) * 100, color="black", linestyle="--", label="同池等权理论参照")
    for axis, label in zip(axes, ("累计净收益 %", "回撤 %")):
        axis.set_ylabel(label, fontproperties=font)
        axis.grid(alpha=.2)
    axes[0].legend(prop=font, ncol=4, loc="upper left")
    axes[0].set_title("十大经典因子 10亿元 100只上限 每5日调仓", fontproperties=font)
    figure.autofmt_xdate()
    figure.tight_layout()
    save(figure, "equity_drawdown.png")
    figure = Figure(figsize=(13, 5), dpi=150)
    axis = figure.subplots()
    ic = diagnostics["ic_summary"].set_index("factor").reindex([f.id for f in FACTORS])
    axis.bar(np.arange(10), ic.rank_ic_mean, color=colors, label="逐日 Rank IC 均值")
    axis.scatter(np.arange(10), ic.rebalance_ic_mean, color="black", marker="D", label="调仓日 Rank IC 均值")
    axis.axhline(0, color="gray", linewidth=.6)
    axis.set_xticks(np.arange(10), [f.name for f in FACTORS], fontproperties=font, rotation=20)
    axis.set_title("因子与随后5日收益的排序相关性", fontproperties=font)
    axis.legend(prop=font)
    axis.grid(axis="y", alpha=.2)
    figure.tight_layout()
    save(figure, "rank_ic.png")
    figure = Figure(figsize=(19, 9), dpi=130)
    axes = figure.subplots(2, 5)
    for axis, factor in zip(axes.flat, FACTORS):
        subset = diagnostics["group_curves"].query("factor == @factor.id")
        for group, frame in subset.groupby("group"):
            axis.plot(pd.to_datetime(frame.date), (frame.unit_nav - 1) * 100, label=f"Q{group}", linewidth=1)
        axis.set_title(factor.name, fontproperties=font)
        axis.grid(alpha=.2)
        axis.tick_params(axis="x", rotation=45, labelsize=7)
    axes.flat[0].legend(fontsize=7, ncol=3)
    figure.suptitle("理论五分组累计收益 % Q5为因子优先方向", fontproperties=font)
    figure.tight_layout()
    save(figure, "group_curves.png")
    figure = Figure(figsize=(10, 6), dpi=150)
    axes = figure.subplots(2, 1, sharex=True)
    coverage = diagnostics["coverage"]
    dates = pd.to_datetime(coverage.date)
    axes[0].plot(dates, coverage.base_eligible, label="基础合格股票")
    axes[0].plot(dates, coverage.common_eligible, label="十因子共同有效股票")
    axes[0].legend(prop=font)
    axes[1].plot(dates, coverage.median_financial_age_days, color="#b36824")
    axes[1].set_ylabel("财报年龄中位数 天", fontproperties=font)
    for axis in axes:
        axis.grid(alpha=.2)
    figure.autofmt_xdate()
    figure.tight_layout()
    save(figure, "coverage.png")
    figure = Figure(figsize=(9, 8), dpi=140)
    axis = figure.subplots()
    matrix = diagnostics["correlation"]
    picture = axis.imshow(matrix, vmin=-1, vmax=1, cmap="RdBu_r")
    labels = [f.name for f in FACTORS]
    axis.set_xticks(range(10), labels, rotation=45, ha="right", fontproperties=font)
    axis.set_yticks(range(10), labels, fontproperties=font)
    for row in range(10):
        for column in range(10):
            axis.text(column, row, f"{matrix[row,column]:.2f}" if np.isfinite(matrix[row,column]) else "—", ha="center", va="center", fontsize=7,
                      color="white" if abs(matrix[row, column]) >= .6 else "black")
    axis.set_title("调仓日截面排名相关性的均值", fontproperties=font)
    figure.colorbar(picture, ax=axis)
    figure.tight_layout()
    save(figure, "factor_correlation.png")


def data_source_section(source_result, audit, market_profile, profile_link):
    """从冻结请求索引与行情清单生成来源、字段、区间和可用时间说明。"""
    manifests = source_result["manifests"]
    def query_dates(kinds):
        return sorted({day for manifest in manifests if manifest["query"]["kind"] in kinds
                       for day in re.findall(r"\d{4}-\d{2}-\d{2}", manifest["query"]["filter"])})
    financial_dates = query_dates({"balance", "income"})
    shares_dates = query_dates({"shares"})
    dividend_dates = query_dates({"dividend"})
    retrieved = sorted(pd.Timestamp(manifest["retrieved_at"]).tz_convert(ZoneInfo("Asia/Seoul")) for manifest in manifests)
    pages = sum(len(manifest["pages"]) for manifest in manifests)
    loaded = market_profile["dates"]
    return [
        "### 3.1 实际数据来源", "",
        "本次回测使用课程公开行情包、东方财富和 BaoStock 三个外部数据渠道。证券范围和实际交易日期由行情矩阵的列与日期行确定。", "",
        "| 数据类别 | 发布方与接口 | 本次使用内容 |", "|---|---|---|",
        "| 日行情、成交与复权 | [课程数据 Release data-backtest-20261016](https://github.com/aslan9/pku_quantllm/releases/tag/data-backtest-20261016) | open、high、low、close、amt、num_trades、cumadj 七个 Feather 矩阵；原始价格用于成交，累计复权因子用于信号与持仓总收益核算 |",
        "| 历史总股本 | 东方财富 RPT_VALUEANALYSIS_DET | TOTAL_SHARES；总市值按本地原始收盘价 × 当日总股本计算；CLOSE_PRICE 用于交叉核对，TOTAL_MARKET_CAP 随响应留存 |",
        "| 一般企业资产负债表 | 东方财富 RPT_F10_FINANCE_GBALANCE | TOTAL_ASSETS 总资产、TOTAL_PARENT_EQUITY 归母净资产 |",
        "| 银行资产负债表 | 东方财富 RPT_F10_FINANCE_BBALANCE | TOTAL_ASSETS、TOTAL_PARENT_EQUITY |",
        "| 保险资产负债表 | 东方财富 RPT_F10_FINANCE_IBALANCE | TOTAL_ASSETS、TOTAL_PARENT_EQUITY |",
        "| 证券公司资产负债表 | 东方财富 RPT_F10_FINANCE_SBALANCE | TOTAL_ASSETS、TOTAL_PARENT_EQUITY |",
        "| 利润表 | 东方财富 RPT_F10_FINANCE_GINCOME | TOTAL_OPERATE_INCOME 营业总收入、PARENT_NETPROFIT 归母净利润 |",
        "| 已实施现金分红 | 东方财富 RPT_SHAREBONUS_DET | ASSIGN_PROGRESS、PRETAX_BONUS_RMB、TOTAL_SHARES、EX_DIVIDEND_DATE、NOTICE_DATE；有更新日期时一并参与可用时间判断 |",
        "| 历史 ST 状态 | BaoStock 0.9.4 / query_daily_history_k_AStock | 使用既有逐日缓存的 isST；原始快照同时保留 tradestatus 等返回字段 |", "",
        f"东方财富统一请求入口为[数据接口]({ENDPOINT})。财报响应同时保留 SECUCODE、REPORT_DATE、NOTICE_DATE、UPDATE_DATE 和 CURRENCY。"
        f"本次抓取于 {retrieved[0]:%Y-%m-%d %H:%M:%S} 至 {retrieved[-1]:%Y-%m-%d %H:%M:%S}（Asia/Seoul，UTC+09:00），"
        f"共 {source_result['requested']:,} 组查询、{pages:,} 页响应。"
        "[sources.json](sources.json)逐项记录接口名称、筛选参数、请求 URL、抓取时间、原始响应路径与 SHA-256；"
        "原始响应保存在 data/classic_factors/raw。", "",
        "### 3.2 数据时间范围与溯源边界", "",
        "| 数据 | 本次范围与用途 |", "|---|---|",
        "| 行情包完整范围 | 2020-01-02 至 2026-10-16，1,643 个交易日 × 5,451 列证券 |",
        f"| 行情实际加载 | {loaded['first_loaded']} 至 {loaded['last_loaded']}，{market_profile['shape'][0]:,} 日 × {market_profile['securities']:,} 列；研究起点之前用于预热；股票范围 {market_profile['stocks']:,} 列 |",
        f"| 历史股本与 ST 状态 | {shares_dates[0]} 至 {shares_dates[-1]}，{len(shares_dates):,} 个决策及交易日 |",
        f"| 财务报告期 | {financial_dates[0]} 至 {financial_dates[-1]}，{len(financial_dates):,} 个季度报告期；匹配本地证券后保留 {audit['financial_records']:,} 条记录 |",
        f"| 分红事件查询 | 除权除息日 {dividend_dates[0]} 至 {dividend_dates[-1]}；因子使用滚动 365 日窗口 |", "",
        "行情包 feathers.zip 的发布 SHA-256 为 `5766950f2a105244867b842709cf4bad890e4e11df0eebeeaa4fec42dc05803d`。"
        f"本地七个输入文件的逐文件哈希、加载区间和维度见[行情输入清单]({profile_link})。"
        "行情可追溯到课程 Release；该发布说明没有披露其底层行情供应商。"
        "历史 ST 缓存位于 data/status/daily，逐日状态按当日开盘前可知推定，覆盖证据见[验收记录](validation.json)。", "",
        "### 3.3 财报字段与历史版本口径", "",
        "财务数据采用东方财富整理的结构化报表字段，币种限定为 CNY，金额按人民币元读取。"
        "本次财务因子使用资产负债表与利润表；现金流量表未参与计算，上市公司原始财报 PDF 未逐份提取校对。"
        "价值使用归母净资产；质量使用归母净利润 TTM 与对应报告期及上年同期平均归母净资产；"
        "成长定义中的营业收入对应原始字段 TOTAL_OPERATE_INCOME，即营业总收入；保守投资使用总资产同比。", "",
        "每条财务记录从 max(REPORT_DATE, NOTICE_DATE, UPDATE_DATE) 日期结束后的下一交易日开始进入信号。"
        "财报按报告期维护各自可用记录，较早报告期的晚修订保留在其自身期间。"
        "TTM 按当期累计值加上年全年值、再减上年同期累计值计算；年报直接使用全年值。"
        "归一化明细保存在 data/classic_factors/financial_records.parquet，source_key 可关联原始请求索引。", "",
        "本研究使用当前抓取版本加保守时间推迟的近似，并非完整历史时点数据库。"
        "源更新时间不能还原全部历史修订过程，后续重算或修订可能改变历史覆盖。"
        "当前行情矩阵的证券范围也可能存在覆盖选择偏差，因此结果适用于本次冻结数据及共同有效样本；"
        "长期有效性仍需更长区间和更完整的历史版本证据。", "",
        "### 3.4 数据覆盖与滞后", "",
    ]


def write_study_report(directory, config, summaries, equities, diagnostics, audit, source_result, study_fingerprint):
    directory = Path(directory)
    summaries.sort(key=lambda item: (-item["sharpe"], -item["annual_return"], item["max_drawdown"], item["factor"]))
    table = pd.DataFrame(summaries)
    columns = ["factor", "factor_name", "return", "annual_return", "net_profit", "ending_nav", "max_drawdown", "annual_volatility", "sharpe", "calmar",
               "benchmark_return", "excess_return_difference", "rank_ic_mean", "rebalance_ic_mean", "icir", "positive_ic_fraction", "average_exposure", "annual_one_way_turnover", "trade_count",
               "total_costs", "capacity_constrained_notional", "ending_market_value", "q5_return", "q1_return", "q5_q1_return_difference", "run_id"]
    table[columns].to_csv(directory / "comparison.csv", index=False, encoding="utf-8-sig")
    periods = period_returns(equities)
    periods.to_csv(directory / "period_returns.csv", index=False, encoding="utf-8-sig")
    combined = pd.concat([frame[["date", "unit_nav"]].assign(factor=factor) for factor, frame in equities.items()], ignore_index=True)
    combined.to_csv(directory / "equity.csv", index=False, encoding="utf-8-sig")
    charts(directory, summaries, equities, diagnostics)
    coverage = diagnostics["coverage"]
    top = summaries[0]
    pct = lambda value: "待有效数据" if value is None or not np.isfinite(float(value)) else f"{float(value):.2%}"
    decimal = lambda value: "待有效数据" if value is None else f"{float(value):.3f}"
    lines = ["# A股十大经典因子策略评估", "",
             f"{config.start} 至 {config.end}，{len(next(iter(equities.values())))} 个交易日。初始资金 10 亿元，最多 100 只股票，95% 目标仓位，每 5 个交易日调仓。",
             "", f"按扣费后夏普排序，{top['factor_name']}位居本样本首位，区间净收益 {pct(top['return'])}，最大回撤 {pct(top['max_drawdown'])}，夏普 {decimal(top['sharpe'])}。平均实际仓位为 {pct(top['average_exposure'])}，应结合下表中的容量和成本阅读。",
             "", "本研究采用免费公开数据的保守探索性口径。财务记录按公告与更新日期中的较晚日期推迟使用，结果同时受到信息滞后、共同样本筛选及日线成交假设的影响。",
             "", "## 1 策略净表现", "", "净收益已计入账户交易费用。年化指标按每年 252 个交易日折算；夏普采用零无风险利率，以日收益均值除以日收益样本标准差后年化。Calmar 为年化收益除以最大回撤，完整指标表同时保留年化波动率和换手率。", "", "| 排名 | 因子 | 区间净收益 | 年化净收益 | 最大回撤 | 夏普 | 平均仓位 | 交易成本 万元 |", "|---:|---|---:|---:|---:|---:|---:|---:|"]
    for rank, item in enumerate(summaries, 1):
        lines.append(f"| {rank} | {item['factor_name']} | {pct(item['return'])} | {pct(item['annual_return'])} | {pct(item['max_drawdown'])} | {decimal(item['sharpe'])} | {pct(item['average_exposure'])} | {float(item['total_costs'])/10000:,.2f} |")
    benchmark_metrics = curve_metrics(diagnostics["benchmark"].unit_nav)
    lines += ["", f"共同股票池等权理论参照的区间收益为 {pct(benchmark_metrics['return'])}。参照每 5 日重配、95% 目标股票仓位，采用零费用、分数份额及充足容量假设；策略表反映实际账户的费用和成交限制。", "",
              "![净值与回撤](equity_drawdown.png)", "", "## 2 因子选股能力", "",
              "Rank IC 是因子排序与随后 5 日收益排序的相关性。全部指标已统一为分数越高越优先；正 IC 表示优先方向与随后收益一致。标签从次日开盘到第六日开盘，最后 6 个决策日的完整标签留空。逐日标签相互重叠，调仓日统计提供对应的五日节奏参照。ICIR 为均值除以标准差。", "",
              "| 因子 | Rank IC 均值 | 调仓日 IC 均值 | ICIR | 正 IC 比例 | 理论 Q5收益 | 理论 Q1收益 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for item in summaries:
        lines.append(f"| {item['factor_name']} | {decimal(item['rank_ic_mean'])} | {decimal(item['rebalance_ic_mean'])} | {decimal(item['icir'])} | {pct(item['positive_ic_fraction'])} | {pct(item['q5_return'])} | {pct(item['q1_return'])} |")
    strongest = max(summaries, key=lambda item: item["rank_ic_mean"] if item["rank_ic_mean"] is not None else -np.inf)
    lines += ["", f"{strongest['factor_name']}的逐日 Rank IC 均值为 {strongest['rank_ic_mean']:.3f}，其理论 Q5 区间收益为 {pct(strongest['q5_return'])}，实际账户区间净收益为 {pct(strongest['return'])}，平均实际仓位为 {pct(strongest['average_exposure'])}。理论分组与实际组合的证券数量、费用和容量条件不同，这些差异应一并用于理解收益。"]
    lines += ["", "Q1 至 Q5 在当日共同股票池内按指标分成五组，Q5 为优先方向。分组曲线使用全部入组证券和理论等权配置，显示因子排序效果；Q5 与 Q1 的差值属于研究诊断。报价中断期间延续持仓及最近有效估值。", "",
              "![排序相关性](rank_ic.png)", "", "![理论五分组](group_curves.png)", "", "![因子相关性](factor_correlation.png)", "",
              "## 3 数据来源与覆盖", ""]
    profile_run = config.path("output_dir") / top["run_id"]
    market_profile = json.loads((profile_run / "data_profile.json").read_text(encoding="utf-8"))
    profile_link = "../../runs/" + top["run_id"] + "/data_profile.json"
    lines += data_source_section(source_result, audit, market_profile, profile_link)
    lines += [
              f"基础合格股票日均 {coverage.base_eligible.mean():,.0f} 只，十因子共同有效股票日均 {coverage.common_eligible.mean():,.0f} 只，范围为 {coverage.common_eligible.min():,} 至 {coverage.common_eligible.max():,} 只。共同有效股票的财报年龄中位数再取日均为 {coverage.median_financial_age_days.mean():.0f} 天，年龄从对应报告期末计算。",
              "", f"累计读取 {audit['financial_records']:,} 条本地证券财务记录，其中 {audit['financial_records_updated_after_notice']:,} 条的更新日期晚于原始公告日期。已实施分红事件共 {audit['dividends']['events']:,} 条，其中 {audit['dividends']['invalid_events']:,} 条的必要字段待核验，其涉及窗口按缺失处理。每项因子的独立覆盖及共同覆盖逐日保存在 [coverage.csv](coverage.csv)。",
              "", "![覆盖与财报年龄](coverage.png)", "", "## 4 成交规则与容量", "",
              "收盘产生固定数量委托，次日原始开盘价复核，卖先买后，执行 T+1 和各板块数量规则。指标同分时按证券代码升序。佣金单边万一、每单最低 5 元，卖出印花税万五，独立滑点万五，另计历史过户费。每证券共享容量为决策日成交额与此前 20 日均成交额较小值的 1%；单次开盘结束后余量到期。名额不足时，每只股票仍以 0.95% 为目标，其余资金保留现金。",
              "", "末日采用 strict_end，在末日开盘按实际规则尝试退出，剩余权益按有效行情估值。选股和退出按逐日信息推进；完整日线的当日成交额用于开盘记录有效性核验，累计复权因子按当日开盘前生效推定。收益采用总收益等价单位，原始价格用于成交和费用。分红用于红利因子的排序指标，总收益等价核算统一处理持仓的公司行动。", "",
              "| 因子 | 年化单边换手 | 容量到期余量累计 亿元 | 期末股票市值 万元 | 成交笔数 |", "|---|---:|---:|---:|---:|"]
    for item in summaries:
        lines.append(f"| {item['factor_name']} | {item['annual_one_way_turnover']:.2f} | {float(item['capacity_constrained_notional'])/1e8:,.2f} | {float(item['ending_market_value'])/1e4:,.2f} | {item['trade_count']:,} |")
    lines += ["", "容量到期余量按各次开盘价折算，含跨日重复尝试和买卖双向委托；该累计量用于解释交易约束。年化单边换手等于逐日双边成交额除以前日净资产后的和，再乘以 252/交易日数/2。", "",
              "交易规则、税费及日历参考以下官方资料。佣金、最低佣金、滑点和成交容量采用[研究配置](config.json)中的参数假设。", ""]
    lines += [f"- [{source['subject']}]({source['url']})" for source in RULE_SOURCES]
    lines += ["", "## 5 分期表现", "",
              "| 因子 | 2025年 | 2026年至9月 |", "|---|---:|---:|"]
    for item in summaries:
        part = periods[(periods.factor == item["factor"]) & (periods.kind == "year")].set_index("period")["return"]
        lines.append(f"| {item['factor_name']} | {pct(part.get('2025'))} | {pct(part.get('2026'))} |")
    lines += ["", "季度收益保存在 [period_returns.csv](period_returns.csv)。窗口、指标和优先方向按固定定义运行，分期结果用于观察本样本的变化。", "", "## 6 因子定义与复现", "",
              "| 因子 | 指标 | 优先方向 |", "|---|---|---|"]
    lines += [f"| {f.name} | {f.formula} | {f.direction} |" for f in FACTORS]
    lines += ["", "营业收入和归母净利润 TTM 按累计财报计算：当期累计值加上年全年值，再减上年同期累计值；年报直接使用全年值。财报按报告期维护各自版本，较早报告期的晚修订保留在其自身期间。红利以已实施方案的每 10 股税前现金金额乘以方案股本再除以 10 计算，观察窗口为 365 日。", "",
              "因子选择参考 [MSCI 八类因子体系](https://www.msci.com/research-and-insights/blog-post/creating-a-common-language-for-factor-investing)、[French 因子数据库](https://mba.tuck.dartmouth.edu/pages/faculty/Ken.French/data_library.html)及[投资因子定义](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/f-f_5_factors_2x3.html)。本研究采用表中可复现的 A 股代表性实现。", "",
              "因子分类与定义参考上述研究来源。策略收益、IC、回撤、理论五分组和共同股票池等权参照由本项目使用第 3 节所列数据计算；夏普的无风险利率设为零。", "",
              f"研究指纹：`{study_fingerprint}`。因子及行情数值计算设备为 Apple MPS，资金与费用采用 Decimal 记账。", "",
              "- [完整指标表](comparison.csv)、[逐日净值](equity.csv)、[IC 明细](rank_ic.csv)、[数据审计](data_audit.json)、[验收记录](validation.json)。",
              "- 批量复现：`.venv/bin/python tools/run_factor_study.py --config configs/classic_factors.toml`。",
              "- 完整重放：在批量命令末尾加入 `--verify-replay`，逐个核对订单、成交、账本、净值与持仓。", "", "逐笔运行目录：", ""]
    for item in summaries:
        relative = "../../runs/" + item["run_id"]
        lines.append(f"- {item['factor_name']}：[单项报告]({relative}/report.md) · [运行配置]({relative}/config.json) · [精确指标]({relative}/summary.json)。")
    (directory / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    atomic_json(directory / "summary.json", clean({"config": config.public(), "study_fingerprint": study_fingerprint,
                 "ranking": summaries, "benchmark": benchmark_metrics,
                 "coverage": {"common_mean": coverage.common_eligible.mean(), "common_min": int(coverage.common_eligible.min()),
                              "common_max": int(coverage.common_eligible.max())}, "data_audit": audit}))
    atomic_json(directory / "manifest.json", {"study_fingerprint": study_fingerprint,
                "source_fingerprint": source_result["fingerprint"],
                "report_generator_sha256": sha256(Path(__file__)),
                "files": {path.name: sha256(path) for path in sorted(directory.iterdir()) if path.is_file() and path.name != "manifest.json"}})
    return summaries
