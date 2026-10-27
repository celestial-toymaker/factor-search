from __future__ import annotations

from datetime import datetime
from importlib.metadata import version
import platform
import sys

import numpy as np

from .config import EXIT_MODES
from .utils import atomic_json, dec, dumps, sha256


def summarize(config, tables, coverage, device, reference, extra):
    equity, trades, orders = tables["equity"], tables["trades"], tables["orders"]
    units = equity.unit_nav.to_numpy(float)
    curve = np.r_[1., units]
    changes = curve[1:] / curve[:-1] - 1
    drawdown = curve / np.maximum.accumulate(curve) - 1
    cost_by_type = {}
    if len(tables["fees"]):
        for kind, frame in tables["fees"].groupby("type"):
            cost_by_type[kind] = str(sum((dec(v) for v in frame.amount), dec(0)))
    last = equity.iloc[-1]
    ledger = extra["ledger"]
    profit = dec(last.nav) - dec(config.initial_cash) - dec(last.net_external_flows)
    total_costs = sum((dec(v) for v in cost_by_type.values()), dec(0))
    summary = {
        "name": config.name, "strategy": config.strategy, "start": config.start, "end": config.end,
        "trading_days": len(equity), "initial_cash": config.initial_cash, "ending_nav": last.nav,
        "net_profit": str(profit), "return": float(units[-1] - 1),
        "annual_return": float(units[-1] ** (252 / len(units)) - 1) if units[-1] > 0 else -1.,
        "max_drawdown": float(-drawdown.min()),
        "sharpe": float(changes.mean() / changes.std(ddof=1) * np.sqrt(252)) if len(changes) > 1 and changes.std(ddof=1) > 0 else 0.,
        "average_exposure": float(equity.exposure.mean()), "ending_exposure": float(last.exposure),
        "target_exposure": float(config.target_exposure), "ending_cash": last.cash,
        "ending_market_value": last.market_value, "ending_positions": int(last.positions),
        "ending_receivables": last.receivables, "ending_pending_equity": last.pending_equity,
        "ending_liabilities": last.liabilities, "stale_value": last.stale_value,
        "first_trade_date": str(trades.date.min()) if len(trades) else None,
        "trade_count": len(trades), "order_count": len(orders),
        "total_costs": str(total_costs), "costs": cost_by_type,
        "turnover_notional": str(sum((dec(v) for v in trades.notional), dec(0))) if len(trades) else "0",
        "expired_orders": int((orders.status == "expired").sum()) if len(orders) else 0,
        "capacity_constrained_orders": int(extra["order_events"].reason.str.contains("容量").sum()) if len(extra["order_events"]) else 0,
        "capacity_constrained_notional": str(sum((dec(value) for value in orders.capacity_unfilled_notional),dec(0))) if len(orders) else "0",
        "exit_mode": config.exit_mode, "exit_label": EXIT_MODES[config.exit_mode],
        "exit_planning": {
            "basis": "actual_path" if config.exit_mode == "latest_feasible" else "daily_rules",
            "uses_future_market_data": config.exit_mode == "latest_feasible",
            "scope": "退出倒排与新增买入的期末退出可行性校验" if config.exit_mode == "latest_feasible" else "当前时点规则",
            "information_boundary": "策略因子、选股与策略回调按当前可获知时间截断；日常撮合按该次开盘规则执行",
            "completed": not ledger.positions and not any(r["amount"] for r in ledger.receivables.values()) and not any(r["quantity"] for r in ledger.pending.values()),
            "disclosure": "本结果的退出规划使用截至期末的后续实际行情倒排，包含新增买入能否按期退出的校验；该功能采用事后规划口径。策略因子、选股、策略回调与日常撮合保持原有时间边界。" if config.exit_mode == "latest_feasible" else "按逐日可获知信息执行所选末日规则。",
        },
        "terminal_assumptions": extra["terminal_settlements"].to_dict("records"),
        "pending_rights": ledger.pending, "receivables": ledger.receivables, "liabilities": ledger.liabilities,
        "accounting_mode": config.accounting_mode, "status_coverage": coverage,
        "compute": device, "created_at": datetime.now().astimezone().isoformat(),
        "warmup": {"first": config.warmup_start, "last_decision": str(extra["universe"].date.iloc[0])},
        "research_assumptions": [
            "当日累计复权因子推定为开盘前生效，使用总收益等价单位；原始价格用于成交和费用。" if config.accounting_mode == "total_return" else "显式公司行动以原始股数、现金、应收款及待上市权益逐项处理。",
            "完整日线的当日成交额用于开盘记录有效性核验；高低收价格用于收盘诊断。",
            "BaoStock 逐日 ST 状态推定在当日开盘前可知；带明确可获知时间的覆盖事件按其时间执行。",
            "价格边界为历史规则与复权参考价推算的候选，首次观测与业务上市事实分别记录。",
            "每次开盘的容量使用其前一决策日成交额与此前 20 日均额；退出模块按这些已实现容量、开盘价和限制倒排，专用后续行情访问包含新增买入退出可行性校验。" if config.exit_mode == "latest_feasible" else "容量仅使用决策日已知成交额；日频开盘成交采用方向性封板规则，盘口排队由更细数据扩展。",
        ],
        "metric_definitions": {"return": "按基金份额法剔除入出金影响的区间收益", "annual_return": "以每年 252 个交易日折算",
                               "max_drawdown": "从此前最高单位净值到随后低点的最大跌幅", "sharpe": "日收益均值除以日收益标准差后年化；无风险利率取零",
                               "net_profit": "终点净资产减初始资金和净入金", "exposure": "股票市值除以净资产；待结算权益单列",
                               "capacity_constrained_notional": "因共享容量到期的委托余量按该次开盘价折算，跨日累计并包含重复尝试，买卖双向合计"},
    }
    if reference is not None:
        summary["benchmark_return"] = float(reference.unit_nav.iloc[-1] - 1)
        summary["benchmark_ending_nav"] = str(reference.nav.iloc[-1])
    return summary


def chart_png(directory, equity, reference=None):
    """以已保存的逐日净值绘制 PNG；中文说明与 Markdown 报告配套。"""
    from pathlib import Path
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.font_manager import FontProperties, fontManager
    from matplotlib.ticker import FuncFormatter
    from matplotlib import dates as mdates

    if equity.empty:
        raise ValueError("资金曲线需要至少一个净值日期")
    dates = np.array(equity.date, dtype="datetime64[D]")
    values = (equity.unit_nav.to_numpy(float) - 1) * 100
    reference_values = None
    if reference is not None:
        reference_values = (reference.set_index("date").unit_nav.reindex(equity.date).to_numpy(float) - 1) * 100
    figure = Figure(figsize=(11.75, 3.625), dpi=160, facecolor="white")
    canvas = FigureCanvasAgg(figure)
    axis = figure.add_subplot(111)
    axis.set_facecolor("white")
    font = None
    mac_font = Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf")
    if mac_font.exists():
        font = FontProperties(fname=str(mac_font), size=9)
    else:
        installed = {record.name for record in fontManager.ttflist}
        for family in ("Noto Sans CJK SC", "PingFang SC", "Microsoft YaHei", "WenQuanYi Zen Hei"):
            if family in installed:
                font = FontProperties(family=family, size=9)
                break
    axis.plot(dates, values, color="#315ee8", linewidth=1.8, label="策略" if font else "Strategy")
    if reference_values is not None:
        axis.plot(dates, reference_values, color="#98a6c0", linewidth=1.3,
                  linestyle=(0, (4, 3)), label="参考组合" if font else "Reference")
    finite = np.r_[0., values[np.isfinite(values)],
                   reference_values[np.isfinite(reference_values)] if reference_values is not None else []]
    lower, upper = float(finite.min()), float(finite.max())
    padding = max((upper - lower) * .12, .5)
    axis.set_ylim(lower - padding, upper + padding)
    axis.axhline(0, color="#dbe1ed", linewidth=.8, zorder=0)
    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}%"))
    ticks = np.unique(np.linspace(0, len(dates) - 1, min(6, len(dates)), dtype=int))
    axis.set_xticks(dates[ticks])
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    labels = axis.get_xticklabels()
    labels[0].set_horizontalalignment("left")
    if len(labels) > 1:
        labels[-1].set_horizontalalignment("right")
    if len(dates) > 1:
        axis.set_xlim(dates[0], dates[-1])
    axis.grid(axis="y", color="#e8edf5", linewidth=.8)
    axis.set_axisbelow(True)
    axis.tick_params(axis="both", labelsize=8, colors="#77849c", length=0, pad=8)
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.legend(loc="upper left", frameon=False, prop=font or FontProperties(size=9), ncol=2)
    figure.subplots_adjust(left=.07, right=.985, bottom=.15, top=.92)
    path = Path(directory) / "equity-curve.png"
    try:
        canvas.print_png(path, metadata={"Software": "ashare_bt"})
    finally:
        figure.clear()
    return path


def write_report(directory, summary, equity, reference=None):
    s = summary
    yuan = lambda value: f"{dec(value):,.2f} 元"
    costs = []
    for key, label, description in [
        ('commission', '佣金', f'单边万分之 {float(s["config"]["commission_rate"])*10000:g}，每单至少 {s["config"]["min_commission"]} 元'),
        ('stamp_tax', '印花税', f'卖出金额的万分之 {float(s["config"]["stamp_tax_rate"])*10000:g}'),
        ('transfer', '过户费', '按历史生效费率'),
        ('slippage', '滑点成本', f'成交金额的万分之 {float(s["config"]["slippage_rate"])*10000:g}，独立计费'),
        ('tax', '显式税款', '依注入金额应计，支付与负债分别记账'),
        ('fee', '显式费用', '依注入的实际金额事件'),
    ]:
        costs.append(f'| {label} | {yuan(s["costs"].get(key,0))} | {description} |')
    assumptions = '\n'.join('- ' + line for line in s['research_assumptions'])
    sources = '\n'.join(f'- [{item["subject"]}]({item["url"]})' for item in s['rule_sources'])
    completion = '已结清持仓及待结算权益' if s['exit_planning']['completed'] else '存在受限持仓或待结算权益，详见账户与退出记录'
    text = f'''# {s['name']} · 中文回测报告

{s['start']} 至 {s['end']}，共 **{s['trading_days']} 个交易日**。初始资金 **{yuan(s['initial_cash'])}**。

## 1. 先看账户赚亏

| 观察项目 | 实际结果 |
|---|---:|
| 投资赚亏 | {yuan(s['net_profit'])} |
| 区间收益率 | {s['return']:.2%} |
| 最大回撤 | {s['max_drawdown']:.2%} |
| 平均实际仓位 | {s['average_exposure']:.2%} |
| 目标仓位 | {s['target_exposure']:.2%} |
| 首笔成交 | {s['first_trade_date'] or '等待合格信号'} |

“投资赚亏”是期末净资产减去初始资金和净入金，已经扣除已计交易成本。“最大回撤”表示资金从此前最高点回落的最大比例。目标仓位表示策略希望投入的比例，实际仓位由成交和账户规则形成。

## 2. 退出方式与后续行情使用范围

**退出方式：{s['exit_label']}。**

{s['exit_planning']['disclosure']}

期末退出状态：**{completion}**。

默认退出逐开盘重放实际价格、逐日规则容量、封板、T+1 与数量规则，倒排最晚启动日。每次开盘容量仍为其前一决策日成交额与此前 20 日均额较小值的 {float(s['config']['participation']):.2%}，同证券累计共享。新增买入按容量允许的最大潜在成交量校验退出可行性，实际买入再由资金和规则决定。后续公告继续按当前可获知时间筛选。该退出功能形成的订单、持仓与收益具有上述事后规划口径。[^exit]

## 3. 资金曲线

![策略与参考组合的累计收益曲线](equity-curve.png)

蓝线表示本次策略，灰色虚线表示首次动量篮子买入持有的参考组合；两者使用相同初始资金、费用和退出方式。收益曲线按份额计量，入出金另行记录。聚合曲线见 [账户净值 CSV](equity.csv) 与 [参考净值 CSV](benchmark_equity.csv)。

## 4. 期末资金与执行结果

| 账户项目 | 金额或数量 |
|---|---:|
| 期末净资产 | {yuan(s['ending_nav'])} |
| 期末现金 | {yuan(s['ending_cash'])} |
| 期末股票及等价持仓市值 | {yuan(s['ending_market_value'])} |
| 应收款 | {yuan(s['ending_receivables'])} |
| 待结算权益 | {yuan(s['ending_pending_equity'])} |
| 负债 | {yuan(s['ending_liabilities'])} |
| 陈旧估值对应市值 | {yuan(s['stale_value'])} |
| 期末持仓证券数 | {s['ending_positions']} |
| 成交笔数 | {s['trade_count']:,} |
| 记录容量限制的订单数 | {s['capacity_constrained_orders']:,} |
| 因容量到期的委托余量累计折合 | {yuan(s['capacity_constrained_notional'])} |

容量受限金额按各次开盘价折算，包含买卖双向及跨日重复尝试；它用于解释执行约束。完整逐笔运行目录中的 `exits.parquet` 保存逐日容量路径、最晚启动日、实际卖出进度和剩余数量。

## 5. 交易成本去向

| 项目 | 金额 | 规则 |
|---|---:|---|
{chr(10).join(costs)}
| 合计 | {yuan(s['total_costs'])} | Decimal 逐笔记账 |

## 6. 数据与研究假设

{assumptions}

历史状态审计覆盖 **{s['status_coverage']['covered_stock_days']:,} / {s['status_coverage']['required_stock_days']:,}** 个待选股票日。预热自 {s['warmup']['first']} 开始，首批决策日为 {s['warmup']['last_decision']}；业绩仅统计上列交易区间。计算设备为 **{s['compute']['device']}**，float32、{s['compute']['block_days']} 日分块。现金、费用与成本采用 Decimal 顺序记账。

## 7. 规则来源与复现

{sources}

- 运行编号：`{s['run_id']}`
- 运行配置：[config.json](config.json)
- 精确金额与指标：[summary.json](summary.json)
- 设备、依赖与源码哈希：[manifest.json](manifest.json)
- 数据指纹：`{s['input_fingerprint']}`
- 状态指纹：`{s['status_fingerprint']}`

本地运行保留订单、成交、借贷分录、持仓、费用、事件、逐日选股与源码快照，可通过工作台或 CLI 追溯单笔成交。

[^exit]: 后续行情访问入口集中在 `src/ashare_bt/exit_path.py`，仅服务退出倒排与新增买入的退出可行性校验；策略历史接口位于 `data.py::History`，日常撮合位于 `execution.py::Broker.match_open`。
'''
    if s['exit_mode'] != 'latest_feasible':
        start = text.index('默认退出逐开盘重放')
        end = text.index('\n\n## 3.', start)
        text = text[:start] + '所选末日模式按逐日信息执行；受限权益或假设清算按本报告记录列示。' + text[end:]
        text = text[:text.index('\n[^exit]:')] + '\n'
    if s['terminal_assumptions']:
        rows = []
        for record in s['terminal_assumptions']:
            rows.append(f"| {record['security']} | {record['type']} | {record['quantity']} | {record['price']} | {yuan(record['gross'])} | {yuan(record['fees'])} | {record['basis']} |")
        detail = '\n\n### 终点假设清算明细\n\n| 证券 | 类型 | 数量 | 估值价格 | 变现金额 | 费用 | 假设依据 |\n|---|---|---:|---:|---:|---:|---|\n' + '\n'.join(rows)
        text = text.replace('\n\n## 3.', detail + '\n\n## 3.', 1)
    if reference is not None:
        reference.to_csv(directory / 'benchmark_equity.csv', index=False, encoding='utf-8-sig')
    else:
        text = text.replace(' 与 [参考净值 CSV](benchmark_equity.csv)', '')
        text = text.replace('蓝线表示本次策略，灰色虚线表示首次动量篮子买入持有的参考组合；两者使用相同初始资金、费用和退出方式。', '蓝线表示本次策略。')
    (directory / 'report.md').write_text(text, encoding='utf-8')
    image_path = chart_png(directory, equity, reference)
    packages = {}
    for name in ("numpy","pandas","pyarrow","torch","fastapi","baostock","matplotlib"):
        packages[name] = version(name)
    source_root = __import__("pathlib").Path(__file__).parent
    manifest = {"python": sys.version, "platform": f"{platform.system()} {platform.release()} {platform.machine()}", "dependencies": packages,
                "report": {"format": "markdown", "document": "report.md", "chart": "equity-curve.png",
                           "chart_format": "png", "chart_size": [1880, 580], "chart_sha256": sha256(image_path)},
                "compute": s["compute"], "exit_planning": s["exit_planning"], "data_fingerprint": s["input_fingerprint"],
                "status_fingerprint": s["status_fingerprint"], "ranges": {"warmup": s["warmup"],"start":s["start"],"end":s["end"],"days":s["trading_days"]},
                "source_sha256": {p.name: sha256(p) for p in sorted(source_root.glob("*.py"))}}
    snapshot = directory / "source"
    snapshot.mkdir(exist_ok=True)
    for path in source_root.glob("*.py"):
        (snapshot / path.name).write_bytes(path.read_bytes())
    upstream = source_root.parents[1] / "research" / "sources" / "manifest.json"
    if upstream.exists():
        import json
        manifest["upstream_sources"] = json.loads(upstream.read_text(encoding="utf-8"))
    atomic_json(directory / "manifest.json", manifest)
