from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, ROUND_FLOOR
from pathlib import Path
from types import SimpleNamespace
import json
import time
import os

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .config import Config
from .compute import compute_features
from .data import History, load_data
from .events import CORPORATE_TYPES, Event, EventSource, at
from .execution import Broker, ExitPlanner
from .ledger import Ledger, ZERO
from .rules import RULE_SOURCES, quantity_rule
from .status import ensure_status
from .strategies import FixedBasketStrategy, StrategyRegistry, TargetWeights, stable_rank
from .utils import atomic_json, check_cancel, dec, dumps, fingerprint, money, sha256

FILTER_REASONS = {0: "合格候选", 1: "证券类型", 2: "行情待观测或异常", 3: "有效历史不足", 4: "ST 状态待核实",
                  5: "ST / *ST 风险警示", 6: "成交额有效性", 7: "流动性门槛", 8: "策略因子准备", 9: "已生效状态限制"}


@dataclass
class BacktestResult:
    config: Config
    equity: pd.DataFrame
    orders: pd.DataFrame
    trades: pd.DataFrame
    journal: pd.DataFrame
    positions: pd.DataFrame
    events: pd.DataFrame
    exits: pd.DataFrame
    fees: pd.DataFrame
    summary: dict
    run_dir: Path | None = None
    benchmark: pd.DataFrame | None = None
    selection: pd.DataFrame | None = None


def eligibility(data, features, status, config, row, ledger=None):
    count = len(data.securities)
    reasons = np.zeros(count, np.int8)
    known_status = status.values[row].copy()
    adv20 = features.values["adv_20"][row].astype(float)
    threshold = float(config.min_adv20)
    boundary = np.flatnonzero(np.isfinite(adv20) & (np.abs(adv20-threshold) <= max(.1,abs(threshold)*1e-5)))
    if row >= 19 and len(boundary):
        adv20[boundary] = data.arrays["amt"][row-19:row+1,boundary].mean(axis=0)
    blocked = np.zeros(count, bool)
    if ledger:
        for security, override in ledger.status_overrides.items():
            column = data.index.get(security)
            if column is None:
                continue
            if "is_st" in override:
                known_status[column] = int(override["is_st"])
            if any(override.get(key) for key in ("delisted", "suspended", "delist_warning")):
                blocked[column] = True
    conditions = [~data.stock_mask,
                  ~features.flags["valid_ohlc"][row] | ~np.isfinite(data.arrays["cumadj"][row]) | (data.arrays["cumadj"][row] <= 0),
                  features.values["history_valid"][row] != 1,
                  known_status < 0, known_status == 1,
                  ~np.isfinite(data.arrays["amt"][row]) | (data.arrays["amt"][row] <= 0),
                  ~np.isfinite(adv20) | (adv20 < threshold)]
    for code, condition in enumerate(conditions, start=1):
        reasons[(reasons == 0) & condition] = code
    reasons[(reasons == 0) & blocked] = 9
    return reasons == 0, reasons


def status_requirement(data, features, config):
    required = (features.flags["valid_ohlc"] & (features.values["history_valid"] == 1) &
                np.isfinite(data.arrays["cumadj"]) & (data.arrays["cumadj"] > 0) &
                (data.arrays["amt"] > 0) & (features.values["adv_20"] >= float(config.min_adv20)) & data.stock_mask[None, :])
    indices = data.trading_indices(config)
    required[:indices[0] - 1] = False
    return required


def _simulate(config, strategy, data, features, status, source, progress, cancel_file, selection_path=None, collect_selection=False, event_handlers=None):
    indices = data.trading_indices(config)
    initial_row = indices[0] - 1
    ledger = Ledger(config.initial_cash, config.accounting_mode, data.dates[initial_row])
    ledger.handlers.update(event_handlers or {})
    broker = Broker(config, data, features, status, ledger)
    planner = ExitPlanner(config, data, broker, lambda: source.known(at(ledger.day, 'decision')))
    runtime_events, equity_rows, position_rows, universe_rows = [], [], [], []
    selection_frames = []
    diagnostic_rows = []
    writer = None
    decision_weights = {}
    previous_eligible = None
    started = time.monotonic()

    def emit(kind, row, phase, payload=None, security=None):
        event = Event(f"E{len(runtime_events) + 1:09d}", kind, at(data.dates[row], phase), at(data.dates[row], phase),
                      security, payload or {}, "engine")
        runtime_events.append({**event.as_dict(), "payload": dumps(event.payload)})
        previous = row if phase in {"close", "rights", "decision", "settlement", "end"} else row - 1
        previous = max(0, previous)
        eligible, reasons = eligibility(data, features, status, config, previous, ledger)
        known = source.known(event.effective_at)
        history = History(data, features, status, previous, eligible, reasons, ledger.status_overrides, known)
        context = SimpleNamespace(time=event.effective_at, history=history, portfolio=ledger.portfolio(), known_events=known)
        strategy.on_event(event, context)

    def apply_due(row, phase):
        ledger.phase = phase
        for event in source.due(at(data.dates[row], phase)):
            impact = ledger.apply(event)
            runtime_events.append({**event.as_dict(), "payload": dumps({**event.payload, "applied": impact})})
            broker.on_action(event)
            if event.type == "order_cancel":
                broker.cancel(event.payload["order_id"], "注入撤单事件")
            if event.type == "order_replace":
                decision_row = max(0,row - 1) if phase in {"pre_open","open"} else row
                execution_row = row if phase in {"pre_open","open"} else row + 1
                if execution_row < len(data.dates):
                    broker.replace(event.payload["order_id"], event.payload["quantity"], decision_row, execution_row)
            previous = max(0, row - 1) if phase in {"pre_open", "open"} else row
            eligible, reasons = eligibility(data, features, status, config, previous, ledger)
            known = source.known(at(data.dates[row], phase))
            strategy.on_event(event, SimpleNamespace(time=at(data.dates[row], phase), history=History(data, features, status, previous, eligible, reasons, ledger.status_overrides, known),
                                                     portfolio=ledger.portfolio(), known_events=known))

    def record_selection(row, eligible, reasons, scores, weights, rebalanced):
        nonlocal writer
        scores = np.asarray(scores, dtype=float)
        final = eligible & np.isfinite(scores)
        chosen = np.array([security in weights for security in data.securities]) & final
        if selection_path is None and not collect_selection:
            return int(chosen.sum())
        reasons = reasons.copy()
        reasons[eligible & ~np.isfinite(scores)] = 8
        ranked = stable_rank(scores, final, data.securities)
        ranks = np.zeros(len(scores), np.int32)
        ranks[ranked] = np.arange(1, len(ranked) + 1)
        state = History(data, features, status, row, eligible, reasons, ledger.status_overrides).status
        frame = pd.DataFrame({"date": data.dates[row], "security": data.securities,
                              "is_st": state, "eligible": final, "filter_code": reasons,
                              "score": scores, "rank": ranks, "adv20": features.values["adv_20"][row],
                              "selected": chosen,
                              "target_weight": [float(weights.get(security,0)) if chosen[column] else 0. for column,security in enumerate(data.securities)],
                              "strategy_weight": [float(weights.get(security,0)) for security in data.securities],
                              "selection_reason": [
                                  ("本次策略目标，当前资格已复核" if rebalanced else "沿用最近策略目标，当前资格已复核") if chosen[column] else
                                  ("最近策略意图暂受资格约束："+FILTER_REASONS[int(reasons[column])]) if security in weights else
                                  FILTER_REASONS[int(reasons[column])] for column,security in enumerate(data.securities)]})
        if collect_selection:
            selection_frames.append(frame)
        if selection_path is not None:
            table = pa.Table.from_pandas(frame, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(selection_path, table.schema, compression="zstd")
            writer.write_table(table)
        return int(chosen.sum())

    def close_decision(row, initial=False):
        nonlocal decision_weights, previous_eligible
        ledger.day, ledger.phase = data.dates[row], "decision"
        eligible, reasons = eligibility(data, features, status, config, row, ledger)
        history = History(data, features, status, row, eligible, reasons, ledger.status_overrides, source.known(at(data.dates[row], "decision")))
        if previous_eligible is None or not np.array_equal(previous_eligible, eligible):
            before = np.zeros_like(eligible) if previous_eligible is None else previous_eligible
            emit("post_universe_changed", row, "decision", {
                "added": [data.securities[j] for j in np.flatnonzero(eligible & ~before)],
                "removed": [data.securities[j] for j in np.flatnonzero(~eligible & before)],
                "eligible_count": int(eligible.sum())})
        previous_eligible = eligible.copy()
        scores = strategy.score(history) if hasattr(strategy, "score") else features.values["mom_20"][row].copy()
        emit("pre_scheduled", row, "decision", {"schedule": "收盘信号与调仓周期"})
        target = strategy.on_close(history, ledger.portfolio())
        emit("post_scheduled", row, "decision", {"schedule": "收盘信号与调仓周期"})
        if getattr(strategy, "last_score_date", None) == history.date:
            scores = strategy.last_scores.copy()
        if isinstance(target, dict):
            target = TargetWeights(target)
        if target is None:
            target = TargetWeights({}, "等待策略信号", False)
        if target.rebalance:
            decision_weights = {}
            for security, weight in target.weights.items():
                if security not in data.index:
                    raise ValueError("目标证券应位于行情数据：" + security)
                if weight > 0 and eligible[data.index[security]]:
                    decision_weights[security] = weight
        for security in list(ledger.positions):
            column = data.index.get(security)
            override = ledger.status_overrides.get(security, {})
            if column is not None and (int(override.get("is_st",status.values[row,column])) == 1 or override.get("delist_warning")):
                planner.active.add(security)
        planner.update(ledger, row)
        for security in planner.active:
            decision_weights.pop(security, None)
        risk_orders = bool(planner.active & set(ledger.positions))
        next_row = row + 1
        if next_row < len(data.dates) and data.dates[next_row] <= config.end:
            nav = ledger.nav
            if target.rebalance or risk_orders:
                desired = decision_weights if target.rebalance else {
                    security: position.market_value / nav for security, position in ledger.positions.items()
                    if security not in planner.active and nav > 0}
                for security, position in sorted(ledger.positions.items()):
                    if security not in data.index:
                        continue
                    if security in planner.active or security not in desired:
                        sell = position.quantity
                    else:
                        column = data.index[security]
                        price = data.arrays["close"][row, column]
                        if not np.isfinite(price) or price <= 0:
                            continue
                        wanted = (nav * desired[security] / dec(price)).to_integral_value(rounding=ROUND_FLOOR)
                        sell = max(ZERO, position.quantity - wanted)
                        if sell < broker.rule(security).minimum:
                            continue
                    if sell > 0:
                        broker.submit(security, "sell", sell, row, next_row,
                                      "按规则计划退出" if security in planner.active else target.reason)
                for priority, (security, weight) in enumerate(desired.items()):
                    if security in planner.active or not eligible[data.index[security]]:
                        continue
                    column = data.index[security]
                    price = dec(data.arrays["close"][row, column])
                    held = ledger.positions[security].quantity if security in ledger.positions else ZERO
                    wanted = (nav * weight / price).to_integral_value(rounding=ROUND_FLOOR)
                    buy = max(ZERO, wanted - held)
                    rule = broker.rule(security)
                    buy = dec(int(buy) // rule.step * rule.step)
                    if buy >= rule.minimum and planner.allow_buy(security, held + buy, row):
                        broker.submit(security, "buy", buy, row, next_row, target.reason, priority)
                    elif buy >= rule.minimum:
                        emit("buy_filtered_by_exit_plan", row, "decision", {"requested": str(buy), "basis": "新增买入经退出模块实际可卖路径核验"}, security)
        for security, position in list(ledger.positions.items()):
            if security not in decision_weights and position.quantity < 1 and next_row < len(data.dates):
                broker.tail_exits[security] = next_row
        qualified_count = record_selection(row, eligible, reasons, scores, decision_weights, target.rebalance)
        universe_rows.append({"date": data.dates[row], "securities": len(data.securities), "stocks": int(data.stock_mask.sum()),
                              "eligible": int(eligible.sum()), "selected": qualified_count,
                              **{f"filtered_{code}": int((reasons == code).sum()) for code in FILTER_REASONS}})

    def broker_event(kind, record):
        row = data.date_index[ledger.day]
        phase = ledger.phase if ledger.phase in {"pre_open", "open", "close", "rights", "decision", "settlement"} else "settlement"
        emit(kind, row, phase, {key:value for key,value in record.items() if key != "candidate_limits"}, record.get("security"))

    broker.listener = broker_event

    try:
        emit("init_portfolio", initial_row, "initialize", {"initial_cash": config.initial_cash})
        emit("post_system_init", initial_row, "initialize")
        emit("post_user_init", initial_row, "initialize")
        emit("before_strategy_run", initial_row, "initialize")
        emit("warmup_finished", initial_row, "close", {"first_loaded": data.dates[0], "last_warmup": data.dates[initial_row]})
        ledger.phase = "close"
        apply_due(initial_row, "rights")
        close_decision(initial_row, True)
        for number, row in enumerate(indices):
            check_cancel(cancel_file)
            day = data.dates[row]
            ledger.day, ledger.phase = day, "pre_open"
            emit("pre_before_trading", row, "pre_open")
            ledger.unlock(day)
            apply_due(row, "pre_open")
            if config.accounting_mode == "total_return":
                relevant = set(ledger.positions) | {order.security for order in broker.orders if order.status == "accepted"}
                for security in sorted(relevant):
                    column = data.index.get(security)
                    if column is None:
                        continue
                    current = data.arrays["cumadj"][row, column]
                    if not np.isfinite(current) or current <= 0:
                        continue
                    position = ledger.positions.get(security)
                    if position:
                        old_factor = position.factor
                    else:
                        existing = next((o for o in broker.orders if o.security == security and o.status == "accepted"), None)
                        old_factor = existing.factor if existing else dec(current)
                    impact = ledger.apply_factor(security, current)
                    broker.adjust_factor(security, dec(current) / old_factor)
                    if impact:
                        emit("adjustment_factor_change", row, "pre_open", impact, security)
            emit("before_trading", row, "pre_open")
            emit("post_before_trading", row, "pre_open")
            ledger.phase = "open"
            apply_due(row, "open")
            ledger.unlock(day)
            emit("pre_open_auction", row, "open")
            broker.match_open(row)
            emit("open_auction", row, "open")
            emit("post_open_auction", row, "open")
            ledger.phase = "close"
            apply_due(row, "close")
            emit("pre_bar", row, "close")
            from .diagnostics import detect_day
            day_diagnostics = detect_day(data, features, status, row)
            diagnostic_rows.extend(day_diagnostics)
            callback_types = {"first_observed", "quote_interrupted", "quote_resumed", "zero_amount", "invalid_ohlc", "trade_count_diagnostic"}
            for record in day_diagnostics:
                if record["type"] in callback_types:
                    emit(record["type"], row, "close", {"detail": record["detail"], "basis": record["basis"]}, record["security"])
            quotes = {security: float(data.arrays["close"][row, data.index[security]])
                      for security in set(ledger.positions) | {record["security"] for record in ledger.pending.values()}
                      if security in data.index and features.flags["valid_ohlc"][row, data.index[security]]}
            ledger.mark(quotes, day)
            emit("bar", row, "close", {"zero_amount": int(features.flags["zero_amount"][row].sum()),
                                      "factor_changes": int(features.flags["factor_change"][row].sum())})
            emit("post_bar", row, "close")
            ledger.phase = "rights"
            apply_due(row, "rights")
            apply_due(row, "decision")
            emit("pre_after_trading", row, "decision")
            close_decision(row)
            emit("after_trading", row, "decision")
            emit("post_after_trading", row, "decision")
            ledger.phase = "settlement"
            emit("pre_settlement", row, "settlement")
            apply_due(row, "settlement")
            if row == indices[-1] and config.exit_mode == "force_cash":
                broker.force_cash(row)
            ledger.assert_invariants()
            market_value = sum((p.market_value for p in ledger.positions.values()), ZERO)
            equity_rows.append({"date": day, "nav": str(ledger.nav), "unit_nav": float(ledger.unit_nav),
                                "cash": str(ledger.cash), "available_cash": str(ledger.available), "frozen_cash": str(ledger.frozen),
                                "market_value": str(market_value), "exposure": float(market_value / ledger.nav) if ledger.nav > 0 else 0.,
                                "target_exposure": float(config.target_exposure),
                                "target_market_value": str(money(ledger.nav * dec(config.target_exposure))),
                                "positions": len(ledger.positions), "receivables": str(sum((r["amount"] for r in ledger.receivables.values()), ZERO)),
                                "pending_equity": str(sum((ledger.pending_value(r) for r in ledger.pending.values()), ZERO)),
                                "liabilities": str(sum(ledger.liabilities.values(), ZERO)), "net_external_flows": str(ledger.net_external_flows),
                                "unit_count": str(ledger.unit_count),
                                "stale_value": str(sum((p.market_value for p in ledger.positions.values() if p.stale_days), ZERO))})
            position_rows.extend(ledger.position_rows())
            emit("settlement", row, "settlement")
            emit("post_settlement", row, "settlement")
            emit("cash_value", row, "settlement", {"cash": str(ledger.cash), "nav": str(ledger.nav), "unit_nav": str(ledger.unit_nav),
                                                   "unit_count": str(ledger.unit_count)})
            if number % 10 == 0 or number == len(indices) - 1:
                progress(stage="backtest", done=number + 1, total=len(indices), message=f"回测 {day} · {number + 1}/{len(indices)}")
        emit("post_strategy_run", indices[-1], "end", {"seconds": round(time.monotonic() - started, 3)})
    finally:
        if writer:
            writer.close()
    tables = {
        "equity": pd.DataFrame(equity_rows), "orders": pd.DataFrame([order.public() for order in broker.orders]),
        "trades": pd.DataFrame(broker.trades), "journal": pd.DataFrame(ledger.journal), "positions": pd.DataFrame(position_rows),
        "events": pd.DataFrame(runtime_events), "exits": pd.DataFrame(planner.records), "fees": pd.DataFrame(ledger.fees),
    }
    extra = {"universe": pd.DataFrame(universe_rows), "order_events": pd.DataFrame(broker.order_events),
             "terminal_settlements": pd.DataFrame(broker.terminal_settlements), "ledger": ledger,
             "initial_basket": getattr(strategy, "initial_basket", None),
             "selection": pd.concat(selection_frames, ignore_index=True) if selection_frames else None,
             "diagnostics": pd.DataFrame(diagnostic_rows)}
    return tables, extra


def run(config: Config | dict | None = None, strategy=None, *, progress=lambda **kw: None,
        cancel_file=None, data=None, features=None, status=None, output=True, run_id=None, collect_selection=False,
        event_source=None, event_handlers=None):
    config = Config() if config is None else Config.from_dict(config) if isinstance(config, dict) else config
    check_cancel(cancel_file)
    strategy = strategy or StrategyRegistry().create(config)
    source = EventSource(event_source.events) if event_source is not None else EventSource.from_jsonl(config.path("events_file") if config.events_file else None)
    if config.accounting_mode == "total_return" and any(event.type in CORPORATE_TYPES for event in source.events):
        raise ValueError("显式公司行动事件请配合 explicit 口径使用")
    data = data or load_data(config, progress)
    features = features or compute_features(data, config, progress, cancel_file)
    status = status or ensure_status(data, config, progress, cancel_file)
    status = status.with_events(source.events)
    coverage = status.coverage(status_requirement(data, features, config))
    if config.strict_status and coverage["unknown_stock_days"]:
        raise RuntimeError(f"请补齐历史 ST 状态：{coverage['unknown_stock_days']:,} 个可选股股票日待核实；示例：{coverage['unknown_examples'][:3]}")
    identity = fingerprint({"config": config.public(), "data": data.fingerprint, "status": status.fingerprint})[:10]
    run_id = run_id or datetime.now().strftime("%Y%m%d-%H%M%S-") + identity
    run_dir = config.path("output_dir") / run_id if output else None
    if run_dir:
        run_dir.mkdir(parents=True, exist_ok=False)
        atomic_json(run_dir / "config.json", config.public())
        run_profile = {**features.profile,"dates":{"first_loaded":data.dates[0],"last_loaded":data.dates[-1],
                                                   "start":config.start,"end":config.end,"trading_days":len(data.trading_indices(config))},
                       "source_files":data.source_files,"fingerprint":data.fingerprint}
        atomic_json(run_dir / "data_profile.json", run_profile)
        atomic_json(run_dir / "status_coverage.json", coverage)
        with (run_dir / "input_events.jsonl").open("w", encoding="utf-8") as handle:
            for event in source.events:
                handle.write(dumps(event.as_dict()) + "\n")
    tables, extra = _simulate(config, strategy, data, features, status, source, progress, cancel_file,
                              run_dir / "selection.parquet" if run_dir else None, collect_selection, event_handlers)
    reference = None
    if config.benchmark:
        check_cancel(cancel_file)
        reference_config = Config.from_dict({**config.public(), "strategy": "momentum", "parameters": {"window": 20, "allocation": "equal"}}, config.root)
        initial = data.date_index[config.start] - 1
        eligible, reasons = eligibility(data, features, status, reference_config, initial)
        probe = StrategyRegistry().create(reference_config)
        first_target = probe.on_close(History(data, features, status, initial, eligible, reasons), Ledger(config.initial_cash).portfolio())
        benchmark_strategy = FixedBasketStrategy(first_target.weights)
        reference_config.name = "首期动量篮子参考组合"
        reference_key = fingerprint({"config":reference_config.public(),"data":data.fingerprint,"status":status.fingerprint,
                                     "events":[event.as_dict() for event in source.events],
                                     "code":{p.name:sha256(p) for p in Path(__file__).parent.glob("*.py")}})
        cache_dir = config.root / "data" / "benchmark" / reference_key
        cache_ready = cache_dir / "ready.json"
        progress(stage="benchmark", done=0, total=len(tables["equity"]), message="计算相同费用与退出规则的参考组合")
        if cache_ready.exists() and not event_handlers:
            reference_tables = {name:pd.read_parquet(cache_dir/f"{name}.parquet") for name in ("equity","trades","fees","orders")}
        else:
            reference_tables, reference_extra = _simulate(reference_config, benchmark_strategy, data, features, status,
                                                          EventSource(source.events), lambda **kw: None, cancel_file, event_handlers=event_handlers)
            if not event_handlers:
                cache_dir.mkdir(parents=True, exist_ok=True)
                for name in ("equity","trades","fees","orders"):
                    target = cache_dir / f"{name}.parquet"
                    temporary = target.with_name(target.name+f".{os.getpid()}.tmp")
                    reference_tables[name].to_parquet(temporary,index=False)
                    temporary.replace(target)
                atomic_json(cache_ready,{"key":reference_key})
        reference = reference_tables["equity"]
        if run_dir:
            for name in ("equity", "trades", "fees", "orders"):
                reference_tables[name].to_parquet(run_dir / f"benchmark_{name}.parquet", index=False)
    from .analytics import summarize, write_report
    summary = summarize(config, tables, coverage, features.metadata, reference, extra)
    summary.update({"run_id": run_id, "input_fingerprint": data.fingerprint, "status_fingerprint": status.fingerprint,
                    "config": config.public(), "rule_sources": RULE_SOURCES,
                    "filter_reasons": FILTER_REASONS, "schema_version": 1})
    if run_dir:
        progress(stage="report", done=0, total=1, message="生成中文报告与可追溯明细")
        for name, table in {**tables, "universe": extra["universe"], "order_events": extra["order_events"],
                            "terminal_settlements": extra["terminal_settlements"], "diagnostics": extra["diagnostics"]}.items():
            table.to_parquet(run_dir / f"{name}.parquet", index=False)
            if name in {"equity", "trades", "orders", "journal", "fees", "terminal_settlements"}:
                csv = table.copy()
                if "candidate_limits" in csv:
                    csv["candidate_limits"] = csv["candidate_limits"].map(dumps)
                csv.to_csv(run_dir / f"{name}.csv", index=False, encoding="utf-8-sig")
        with (run_dir / "events.jsonl").open("w", encoding="utf-8") as handle:
            for record in tables["events"].to_dict("records"):
                handle.write(dumps({**record, "payload": json.loads(record["payload"])}) + "\n")
        atomic_json(run_dir / "summary.json", summary)
        write_report(run_dir, summary, tables["equity"], reference)
        progress(stage="report", done=1, total=1, message="报告已就绪")
    return BacktestResult(config=config, **tables, summary=summary, run_dir=run_dir, benchmark=reference, selection=extra["selection"])
