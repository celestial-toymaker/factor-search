"""因子 IC、理论分组组合、容量解释及中文比较报告。"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch

from .classic_factors import FACTORS
from .compute import select_device
from .engine import eligibility
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
    updated = clean({**summary, **extra, "factor_definition": asdict(factor)})
    atomic_json(Path(directory) / "summary.json", updated)
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
            axis.text(column, row, f"{matrix[row,column]:.2f}" if np.isfinite(matrix[row,column]) else "—", ha="center", va="center", fontsize=7)
    axis.set_title("调仓日截面排名相关性的均值", fontproperties=font)
    figure.colorbar(picture, ax=axis)
    figure.tight_layout()
    save(figure, "factor_correlation.png")


def write_study_report(directory, config, summaries, equities, diagnostics, audit, source_result, study_fingerprint):
    directory = Path(directory)
    summaries.sort(key=lambda item: (-item["sharpe"], -item["annual_return"], item["max_drawdown"], item["factor"]))
    table = pd.DataFrame(summaries)
    columns = ["factor", "factor_name", "return", "annual_return", "max_drawdown", "annual_volatility", "sharpe", "calmar",
               "rank_ic_mean", "icir", "positive_ic_fraction", "average_exposure", "annual_one_way_turnover", "trade_count",
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
             "", "## 1 策略净表现", "", "| 排名 | 因子 | 区间净收益 | 年化净收益 | 最大回撤 | 夏普 | 平均仓位 | 交易成本 万元 |", "|---:|---|---:|---:|---:|---:|---:|---:|"]
    for rank, item in enumerate(summaries, 1):
        lines.append(f"| {rank} | {item['factor_name']} | {pct(item['return'])} | {pct(item['annual_return'])} | {pct(item['max_drawdown'])} | {decimal(item['sharpe'])} | {pct(item['average_exposure'])} | {float(item['total_costs'])/10000:,.2f} |")
    benchmark_metrics = curve_metrics(diagnostics["benchmark"].unit_nav)
    lines += ["", f"共同股票池等权理论参照的区间收益为 {pct(benchmark_metrics['return'])}。参照每 5 日重配、95% 目标股票仓位，采用零费用、分数份额及充足容量假设；策略表反映实际账户的费用和成交限制。", "",
              "![净值与回撤](equity_drawdown.png)", "", "## 2 因子选股能力", "",
              "Rank IC 是因子排序与随后 5 日收益排序的相关性。全部指标已统一为分数越高越优先；正 IC 表示优先方向与随后收益一致。标签从次日开盘到第六日开盘，最后 6 个决策日的完整标签留空。逐日标签相互重叠，调仓日统计提供对应的五日节奏参照。ICIR 为均值除以标准差。", "",
              "| 因子 | Rank IC 均值 | 调仓日 IC 均值 | ICIR | 正 IC 比例 | 理论 Q5收益 | 理论 Q1收益 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for item in summaries:
        lines.append(f"| {item['factor_name']} | {decimal(item['rank_ic_mean'])} | {decimal(item['rebalance_ic_mean'])} | {decimal(item['icir'])} | {pct(item['positive_ic_fraction'])} | {pct(item['q5_return'])} | {pct(item['q1_return'])} |")
    lines += ["", "Q1 至 Q5 在当日共同股票池内按指标分成五组，Q5 为优先方向。分组曲线使用全部入组证券和理论等权配置，显示因子排序效果；Q5 与 Q1 的差值属于研究诊断。报价中断期间延续持仓及最近有效估值。", "",
              "![排序相关性](rank_ic.png)", "", "![理论五分组](group_curves.png)", "", "![因子相关性](factor_correlation.png)", "",
              "## 3 数据覆盖与滞后", "",
              f"基础合格股票日均 {coverage.base_eligible.mean():,.0f} 只，十因子共同有效股票日均 {coverage.common_eligible.mean():,.0f} 只，范围为 {coverage.common_eligible.min():,} 至 {coverage.common_eligible.max():,} 只。共同有效股票的财报年龄中位数再取日均为 {coverage.median_financial_age_days.mean():.0f} 天，年龄从对应报告期末计算。",
              "", f"累计读取 {audit['financial_records']:,} 条本地证券财务记录，其中 {audit['financial_records_updated_after_notice']:,} 条的更新日期晚于原始公告日期。时点或币种待核验而排除 {audit['rejected_financial_records']:,} 条。每项因子的独立覆盖及共同覆盖逐日保存在 [coverage.csv](coverage.csv)。",
              "", "历史版本采用源数据更新时间的保守近似，源数据在后续重算或修订时可能改变历史覆盖。当前行情矩阵的证券范围也可能存在覆盖选择偏差。因此结果适用于本次冻结数据及共同有效样本，长期有效性仍需更长区间和更完整的历史版本证据。",
              "", "![覆盖与财报年龄](coverage.png)", "", "## 4 成交规则与容量", "",
              "收盘产生固定数量委托，次日原始开盘价复核，卖先买后，执行 T+1 和各板块数量规则。佣金单边万一、每单最低 5 元，卖出印花税万五，独立滑点万五，另计历史过户费。每证券共享容量为决策日成交额与此前 20 日均成交额较小值的 1%；单次开盘结束后余量到期。名额不足时，每只股票仍以 0.95% 为目标，其余资金保留现金。",
              "", "末日采用 strict_end，在末日开盘按实际规则尝试退出，剩余权益按有效行情估值。选股和退出按逐日信息推进；完整日线的当日成交额用于开盘记录有效性核验，累计复权因子按当日开盘前生效推定。收益采用总收益等价单位，原始价格用于成交和费用。分红用于红利因子的排序指标，总收益等价核算统一处理持仓的公司行动。", "",
              "| 因子 | 年化单边换手 | 容量到期余量累计 亿元 | 期末股票市值 万元 | 成交笔数 |", "|---|---:|---:|---:|---:|"]
    for item in summaries:
        lines.append(f"| {item['factor_name']} | {item['annual_one_way_turnover']:.2f} | {float(item['capacity_constrained_notional'])/1e8:,.2f} | {float(item['ending_market_value'])/1e4:,.2f} | {item['trade_count']:,} |")
    lines += ["", "容量到期余量按各次开盘价折算，含跨日重复尝试和买卖双向委托；该累计量用于解释交易约束。年化单边换手等于逐日双边成交额除以前日净资产后的和，再乘以 252/交易日数/2。", "", "## 5 分期表现", "",
              "| 因子 | 2025年 | 2026年至9月 |", "|---|---:|---:|"]
    for item in summaries:
        part = periods[(periods.factor == item["factor"]) & (periods.kind == "year")].set_index("period")["return"]
        lines.append(f"| {item['factor_name']} | {pct(part.get('2025'))} | {pct(part.get('2026'))} |")
    lines += ["", "季度收益保存在 [period_returns.csv](period_returns.csv)。窗口、指标和优先方向按固定定义运行，分期结果用于观察本样本的变化。", "", "## 6 因子定义与复现", "",
              "| 因子 | 指标 | 优先方向 |", "|---|---|---|"]
    lines += [f"| {f.name} | {f.formula} | {f.direction} |" for f in FACTORS]
    lines += ["", "营业收入和归母净利润 TTM 按累计财报计算：当期累计值加上年全年值，再减上年同期累计值；年报直接使用全年值。财报按报告期维护各自版本，较早报告期的晚修订保留在其自身期间。红利以已实施方案的每 10 股税前现金金额乘以方案股本再除以 10 计算，观察窗口为 365 日。", "",
              "因子选择参考 [MSCI 八类因子体系](https://www.msci.com/research-and-insights/blog-post/creating-a-common-language-for-factor-investing)、[French 因子数据库](https://mba.tuck.dartmouth.edu/pages/faculty/Ken.French/data_library.html)及[投资因子定义](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/f-f_5_factors_2x3.html)。本研究采用表中可复现的 A 股代表性实现。", "",
              "行情来自项目本地课程矩阵；历史 ST 状态来自已有 BaoStock 缓存。财务、市值和分红取自[东方财富数据中心](https://data.eastmoney.com/)，原始响应、请求参数、抓取时间和哈希保存在本地数据缓存，索引见 [sources.json](sources.json)。", "",
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
                "files": {path.name: sha256(path) for path in sorted(directory.iterdir()) if path.is_file() and path.name != "manifest.json"}})
    return summaries
