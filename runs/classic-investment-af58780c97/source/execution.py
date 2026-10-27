from __future__ import annotations

from dataclasses import asdict, dataclass
from decimal import Decimal, ROUND_FLOOR
import math

import numpy as np

from .ledger import Ledger, ZERO
from .rules import FeeModel, candidate_limits, legal_quantity, quantity_rule, split_quantity, tick
from .utils import dec, dumps, money

TRANSITIONS = {
    "submitted": {"accepted", "rejected"}, "accepted": {"partially_filled", "filled", "cancelled", "expired"},
    "partially_filled": {"partially_filled", "filled", "cancelled", "expired"},
    "filled": set(), "rejected": set(), "cancelled": set(), "expired": set(),
}


@dataclass
class Order:
    order_id: str
    intent_id: str
    security: str
    side: str
    quantity: Decimal
    decision_index: int
    execution_index: int
    decision_date: str
    execution_date: str
    reference_price: Decimal
    factor: Decimal
    reason: str
    priority: int = 0
    status: str = "submitted"
    filled: Decimal = ZERO
    reserved_cash: Decimal = ZERO
    reserved_quantity: Decimal = ZERO
    commission_paid: Decimal = ZERO
    notional: Decimal = ZERO
    last_reason: str = ""
    capacity_unfilled_notional: Decimal = ZERO

    @property
    def remaining(self):
        return self.quantity - self.filled

    def public(self):
        result = asdict(self)
        for key, value in result.items():
            if isinstance(value, Decimal):
                result[key] = str(value)
        result["remaining"] = str(self.remaining)
        return result


class Broker:
    def __init__(self, config, data, features, status, ledger: Ledger):
        self.config, self.data, self.features, self.status, self.ledger = config, data, features, status, ledger
        self.fee_model = FeeModel(config)
        self.orders: list[Order] = []
        self.order_events = []
        self.trades = []
        self.terminal_settlements = []
        self.tail_exits = {}
        self.capacity_used = {}
        self._intent_counter = 0
        self.listener = None

    def notify(self, kind, record):
        if self.listener:
            self.listener(kind, record)

    def rule(self, security):
        return quantity_rule(security, {**self.ledger.rules.get("market", {}), **self.ledger.rules.get(security, {})})

    def transition(self, order, state, reason=""):
        if state not in TRANSITIONS[order.status]:
            raise ValueError(f"订单状态转换需要有效路径：{order.status} → {state}")
        previous = order.status
        order.status = state
        order.last_reason = reason
        self.order_events.append({"event_id": f"OE{len(self.order_events) + 1:08d}", "date": self.ledger.day,
                                  "phase": self.ledger.phase, "order_id": order.order_id, "security": order.security,
                                  "from": previous, "to": state, "reason": reason,
                                  "filled": str(order.filled), "remaining": str(order.remaining)})
        self.notify("order_" + state, self.order_events[-1])

    def capacity(self, row, column):
        recent = self.data.arrays["amt"][max(0, row - 19):row + 1, column]
        if len(recent) < 20 or not np.isfinite(recent).all() or np.any(recent < 0):
            return ZERO
        average = sum((dec(value) for value in recent), ZERO) / 20
        return (min(dec(recent[-1]), average) * dec(self.config.participation)).quantize(Decimal("0.01"), rounding=ROUND_FLOOR)

    def submit(self, security, side, quantity, row, execution_row, reason="策略目标", priority=0):
        if side not in {"buy", "sell"} or dec(quantity) <= 0:
            raise ValueError("委托需要有效买卖方向和正数量")
        if security not in self.data.index:
            raise ValueError("委托证券应位于数据集合：" + security)
        column = self.data.index[security]
        price = self.data.arrays["close"][row, column]
        if not np.isfinite(price) or price <= 0:
            _, price = self.data.previous("close", row + 1, column)
        if price is None:
            raise ValueError("委托需要决策时已知的参考价格")
        factor = self.data.arrays["cumadj"][row, column]
        factor = dec(factor) if np.isfinite(factor) and factor > 0 else Decimal(1)
        self._intent_counter += 1
        intent = f"I{self._intent_counter:08d}"
        submitted = []
        for piece in split_quantity(security, quantity, side, self.rule(security)):
            order = Order(f"O{len(self.orders) + 1:08d}", intent, security, side, dec(piece), row, execution_row,
                          self.data.dates[row], self.data.dates[execution_row], dec(price), factor, reason, priority)
            self.orders.append(order)
            self.ledger.source_event = order.order_id
            self.order_events.append({"event_id": f"OE{len(self.order_events) + 1:08d}", "date": self.ledger.day,
                                      "phase": self.ledger.phase, "order_id": order.order_id, "security": security,
                                      "from": "created", "to": "submitted", "reason": "策略委托提交校验", "filled": "0", "remaining": str(order.remaining)})
            self.notify("order_submitted", self.order_events[-1])
            if side == "sell":
                position = self.ledger.positions.get(security)
                if position is None:
                    self.transition(order, "rejected", "证券持仓已结清")
                    continue
                reserved = min(order.quantity, position.sellable(order.execution_date))
                if reserved < order.quantity:
                    self.transition(order, "rejected", "执行日预计可卖数量不足")
                    continue
                order.reserved_quantity = reserved
                position.reserved += reserved
            else:
                fees = self.fee_model.quote(security, order.execution_date, side, order.quantity * order.reference_price)
                estimate = order.quantity * order.reference_price + sum(fees.values(), ZERO)
                order.reserved_cash = self.ledger.reserve_cash(estimate)
            self.transition(order, "accepted", "已登记，开盘复核资金与规则")
            submitted.append(order)
        if side == "sell" and dec(quantity) < 1:
            self.tail_exits[security] = execution_row
        return submitted

    def release(self, order):
        self.ledger.source_event = order.order_id
        if order.reserved_cash:
            self.ledger.release_cash(order.reserved_cash)
            order.reserved_cash = ZERO
        if order.reserved_quantity:
            position = self.ledger.positions.get(order.security)
            order.reserved_quantity = ZERO
            if position:
                position.reserved = sum((pending.reserved_quantity for pending in self.orders
                                         if pending.security == order.security), ZERO)

    def cancel(self, order_id, reason="撤销委托"):
        order = next((order for order in self.orders if order.order_id == order_id), None)
        if order is None or order.status not in {"accepted", "partially_filled"}:
            self.order_events.append({"event_id": f"OE{len(self.order_events) + 1:08d}", "date": self.ledger.day,
                                      "phase": self.ledger.phase, "order_id": order_id, "security": order.security if order else None,
                                      "from": order.status if order else "unknown", "to": "cancellation_rejected", "reason": "撤单资格复核：订单编号或当前终态",
                                      "filled": str(order.filled) if order else "0", "remaining": str(order.remaining) if order else "0"})
            self.notify("order_cancellation_rejected", self.order_events[-1])
            return False
        self.release(order)
        self.transition(order, "cancelled", reason)
        return True

    def replace(self, order_id, quantity, row, execution_row):
        old = next((order for order in self.orders if order.order_id == order_id), None)
        if old is None or not self.cancel(order_id, "修改：撤销旧单后重建"):
            return []
        return self.submit(old.security, old.side, quantity, row, execution_row, "修改自 " + order_id, old.priority)

    def on_action(self, event):
        if event.type == "split":
            ratio = dec(event.payload["ratio"])
            for order in self.orders:
                if order.security == event.security and order.status == "accepted":
                    order.quantity = dec(legal_quantity(order.security, order.quantity * ratio, order.side,
                                                       order.quantity * ratio if order.side == "sell" else None, self.rule(order.security)))
                    order.reference_price /= ratio
                    order.reserved_quantity *= ratio
        elif event.type == "code_change":
            for order in self.orders:
                if order.security == event.security and order.status == "accepted":
                    order.security = event.payload["target_security"]
                    if order.security not in self.data.index:
                        self.cancel(order.order_id,"代码转换后的证券需要补充行情与可交易日期")
        elif event.type in {"conversion", "merger", "delist_settlement"}:
            for order in self.orders:
                if order.security == event.security and order.status in {"accepted", "partially_filled"}:
                    self.cancel(order.order_id, "公司行动结束原证券委托")

    def adjust_factor(self, security, ratio):
        for order in self.orders:
            if order.security != security or order.status != "accepted":
                continue
            order.reserved_quantity *= ratio
            if abs(ratio - 1) >= Decimal("0.0001"):
                order.quantity = dec(legal_quantity(security, order.quantity * ratio, order.side,
                                                   order.quantity * ratio if order.side == "sell" else None, self.rule(security)))
                order.reference_price /= ratio
            order.factor *= ratio

    def opening_context(self, row, security):
        column = self.data.index[security]
        raw_price = self.data.arrays["open"][row, column]
        amount = self.data.arrays["amt"][row, column]
        factor = self.data.arrays["cumadj"][row, column]
        prior_row, previous_close = self.data.previous("close", row, column)
        previous_factor = self.data.arrays["cumadj"][prior_row, column] if prior_row is not None else 1
        if not np.isfinite(previous_factor) or previous_factor <= 0:
            previous_factor = 1
        if not np.isfinite(factor) or factor <= 0:
            factor = previous_factor
        overrides = self.ledger.status_overrides.get(security, {})
        rules = {**self.ledger.rules.get("market", {}), **self.ledger.rules.get(security, {})}
        flag = int(overrides.get("is_st", self.status.open_values[row, column]))
        bands = candidate_limits(security, self.data.dates[row], previous_close, previous_factor, factor,
                                 flag == 1 if flag >= 0 else None,
                                 rules.get("unlimited", False), rules.get("limit_rate"))
        price = tick(raw_price) if np.isfinite(raw_price) and raw_price > 0 else None
        return column, price, amount, dec(factor), flag, bands, overrides

    def block_reason(self, row, order, context):
        _, price, amount, _, flag, bands, overrides = context
        if price is None:
            return "开盘报价缺失或异常"
        if not np.isfinite(amount) or amount <= 0:
            return "当日成交额有效性核验：零成交额或缺失；采用完整日线研究假设"
        current_factor = self.data.arrays["cumadj"][row, self.data.index[order.security]]
        if self.config.accounting_mode == "total_return" and (not np.isfinite(current_factor) or current_factor <= 0):
            return "总收益等价单位的当日复权因子待核实"
        if overrides.get("delisted") or overrides.get("suspended"):
            return "已生效证券交易状态限制"
        if order.side == "buy" and (flag != 0 or overrides.get("delist_warning")):
            return "买入资格复核：需要已确认的非 ST 状态"
        boundary = "upper" if order.side == "buy" else "lower"
        if any(price == dec(band[boundary]) for band in bands):
            return "开盘处于候选涨停价，买入零成交" if order.side == "buy" else "开盘处于候选跌停价，卖出零成交"
        return ""

    def match_open(self, row):
        day = self.data.dates[row]
        active = [order for order in self.orders if order.execution_index == row and order.status == "accepted"]
        active.sort(key=lambda order: (order.side == "buy", order.priority, order.security, order.order_id))
        caps = {}
        for order in active:
            cap = self.capacity(order.decision_index, self.data.index[order.security])
            caps[order.security] = min(caps.get(order.security, cap), cap)
        used = {}
        for order in active:
            self.release(order)
            self.ledger.source_event = order.order_id
            context = self.opening_context(row, order.security)
            column, price, amount, factor, flag, bands, overrides = context
            reason = self.block_reason(row, order, context)
            if not reason and order.side == "buy" and order.security not in self.ledger.positions and len(self.ledger.positions) >= self.config.max_positions:
                reason = "退出受限持仓占用名额，等待持仓名额释放"
            remaining_cap = max(ZERO, caps[order.security] - used.get(order.security, ZERO))
            if not reason:
                position = self.ledger.positions.get(order.security)
                requested = min(order.remaining, (remaining_cap / price).to_integral_value(rounding=ROUND_FLOOR))
                if order.side == "sell":
                    requested = min(requested, position.sellable(day) if position else ZERO)
                quantity = legal_quantity(order.security, requested, order.side, position.quantity if position and order.side == "sell" else None, self.rule(order.security))
                if order.side == "buy":
                    quantity = min(quantity, legal_quantity(order.security, self.ledger.available / price, "buy", rule=self.rule(order.security)))
                    while quantity:
                        fees = self.fee_model.quote(order.security, day, order.side, dec(quantity) * price, order.notional, order.commission_paid)
                        total = dec(quantity) * price + sum(fees.values(), ZERO)
                        if total <= self.ledger.available:
                            break
                        rule = self.rule(order.security)
                        short = total - self.ledger.available
                        step_reduction = max(rule.step, math.ceil(float(short / price) / rule.step) * rule.step)
                        quantity = legal_quantity(order.security, quantity - step_reduction, "buy", rule=rule)
                if quantity and order.side == "sell":
                    prospective_fees = self.fee_model.quote(order.security, day, order.side, dec(quantity) * price, order.notional, order.commission_paid)
                    if dec(quantity) * price + self.ledger.available < sum(prospective_fees.values(), ZERO):
                        quantity = 0
                        reason = "成交所得与可用现金不足以支付费用"
                if quantity:
                    trade_id = f"T{len(self.trades) + 1:08d}"
                    gross = money(dec(quantity) * price)
                    fees = self.fee_model.quote(order.security, day, order.side, gross, order.notional, order.commission_paid)
                    if order.side == "buy":
                        unlock_date = self.data.dates[row + 1] if row + 1 < len(self.data.dates) else "9999-12-31"
                        self.ledger.buy(order.security, quantity, price, fees, unlock_date, factor, order.order_id, trade_id)
                        impact = {"gross": gross, "cost": gross, "proxy_fraction_cash": ZERO, "net_realized_pnl": ZERO}
                    else:
                        impact = self.ledger.sell(order.security, quantity, price, fees, order.order_id, trade_id)
                    order.filled += dec(quantity)
                    order.notional += gross
                    order.commission_paid += fees["commission"]
                    used[order.security] = used.get(order.security, ZERO) + gross
                    self.trades.append({"trade_id": trade_id, "order_id": order.order_id, "intent_id": order.intent_id,
                                        "date": day, "decision_date": order.decision_date, "security": order.security,
                                        "side": order.side, "quantity": str(quantity), "price": str(price),
                                        "notional": str(gross), "fees": str(sum(fees.values(), ZERO)),
                                        **{key: str(value) for key, value in fees.items()},
                                        "net_realized_pnl": str(impact["net_realized_pnl"]),
                                        "proxy_fraction_cash": str(impact["proxy_fraction_cash"]),
                                        "cash_after": str(self.ledger.cash), "reason": order.reason,
                                        "capacity": str(caps[order.security]), "capacity_used": str(used[order.security]),
                                        "candidate_limits": bands, "execution_basis": "次日原始开盘价；完整日线成交额有效性核验",
                                        "kind": "exchange_simulation"})
                    self.transition(order, "filled" if order.remaining == 0 else "partially_filled", "开盘成交")
                    self.notify("trade", self.trades[-1])
                if order.remaining and not reason:
                    capacity_left = max(ZERO, caps[order.security] - used.get(order.security, ZERO))
                    if capacity_left < price * self.rule(order.security).minimum:
                        reason = "共享成交容量不足或低于合法数量"
                    elif order.side == "sell":
                        reason = "可卖持仓或数量规则限制"
                    else:
                        reason = "可用资金、费用或数量规则限制"
            if order.status in {"accepted", "partially_filled"}:
                if "容量" in reason:
                    order.capacity_unfilled_notional = money(order.remaining * (price or order.reference_price))
                self.transition(order, "expired", reason or "单次开盘撮合结束，余量到期")
        self.capacity_used[day] = {key: str(value) for key, value in used.items()}
        for security, execution in list(self.tail_exits.items()):
            if execution != row:
                continue
            position = self.ledger.positions.get(security)
            if position and position.quantity < 1 and position.sellable(day) == position.quantity:
                context = self.opening_context(row, security)
                dummy = type("Tail", (), {"side": "sell", "security": security})()
                if not self.block_reason(row, dummy, context):
                    quantity, price = position.quantity, context[1]
                    self.ledger.source_event = "proxy-tail:" + security
                    self.ledger.sell(security, quantity, price, {})
            self.tail_exits.pop(security)

    def force_cash(self, row):
        self.ledger.phase = "terminal_assumption"
        for order in self.orders:
            if order.status in {"accepted", "partially_filled"}:
                self.cancel(order.order_id, "终点清算释放委托占用")
        for security, position in list(self.ledger.positions.items()):
            price = position.last_price
            if price <= 0:
                raise ValueError("终点假设清算需要明确估值：" + security)
            quantity = position.quantity
            fees = self.fee_model.quote(security, self.ledger.day, "sell", quantity * price)
            position.lots = []
            position.reserved = ZERO
            self.ledger.source_event = "terminal:" + security
            impact = self.ledger.sell(security, quantity, price, fees, "terminal", "terminal", allow_fee_liability=True)
            self.terminal_settlements.append({"date": self.ledger.day, "security": security, "quantity": str(quantity),
                                               "price": str(price), "gross": str(impact["gross"]),
                                               "fees": str(sum(fees.values(), ZERO)), "type": "终点假设变现",
                                               "basis": "末日收盘或最近有效估值；容量、封板、停牌与锁定限制在此清算假设中单列"})
        for key, record in list(self.ledger.pending.items()):
            amount = self.ledger.pending_value(record)
            self.ledger.source_event = "terminal-pending:" + key
            self.ledger.post("terminal_pending_settlement", {"cash:available": amount, f"asset:pending:{key}": -record["cost"],
                                                             "income:realized": record["cost"] - amount}, record["security"])
            fees = self.fee_model.quote(record["security"], self.ledger.day, "sell", amount)
            self.ledger.charge_fees(fees, record["security"], "terminal", "terminal-pending", allow_liability=True)
            self.terminal_settlements.append({"date": self.ledger.day, "security": record["security"],
                                               "quantity": str(record["quantity"]), "price": str(record["valuation"]),
                                               "gross": str(amount), "fees": str(sum(fees.values(), ZERO)), "type": "待结算权益假设变现",
                                               "basis": "显式提供的权益估值或最近有效报价"})
            self.ledger.pending.pop(key)
        for key in list(self.ledger.receivables):
            self.ledger.source_event = "terminal-receivable:" + key
            self.ledger.pay_receivable(key)
        self.ledger.pay_liabilities()


class ExitPlanner:
    def __init__(self, config, data, broker, known_events=lambda: ()):
        self.config, self.data, self.broker = config, data, broker
        self.active = set()
        self.records = []
        self.started = {}
        self.completed = set()
        from .exit_path import ActualExitPath
        self.actual_path = ActualExitPath(broker, known_events)

    def estimate(self, security, quantity, row, newly_bought=False):
        column = self.data.index.get(security)
        capacity = self.broker.capacity(row, column) if column is not None else ZERO
        remaining_sessions = sum(self.data.dates[index] <= self.config.end for index in range(row + 1, len(self.data.dates)))
        if self.config.exit_mode == 'latest_feasible':
            from copy import deepcopy
            from .ledger import Position
            position = deepcopy(self.broker.ledger.positions.get(security, Position(security)))
            position.quantity = dec(quantity)
            if security not in self.broker.ledger.positions and column is not None:
                factor = self.data.arrays['cumadj'][row, column]
                position.factor = dec(factor) if np.isfinite(factor) and factor > 0 else Decimal(1)
            path = self.actual_path.plan(security, position, row)
        else:
            path = {'feasible': None, 'latest_start': None, 'required_sessions': None, 'fills': [], 'terminal_quantity': str(quantity)}
        return {"date": self.data.dates[row], "security": security, "quantity": str(quantity), "known_capacity": str(capacity),
                "required_sessions": path['required_sessions'], 'actual_latest_start': path['latest_start'],
                "remaining_sessions": remaining_sessions, 'path_feasible': path['feasible'],
                'projected_terminal_quantity': path['terminal_quantity'], 'capacity_path': dumps(path['fills']),
                'uses_future_market_data': self.config.exit_mode == 'latest_feasible',
                "basis": "退出模块专用后续行情；逐开盘重放实际价格、规则容量、封板、T+1 和数量规则；公告按当前可获知时间截断" if self.config.exit_mode == 'latest_feasible' else '末日按所选规则处理'}

    def update(self, ledger, row):
        for security, position in sorted(ledger.positions.items()):
            estimate = self.estimate(security, position.quantity, row)
            following = self.data.dates[row + 1] if row + 1 < len(self.data.dates) else None
            due = estimate['path_feasible'] is False or (following and estimate['actual_latest_start'] and estimate['actual_latest_start'] <= following)
            if self.config.exit_mode == "latest_feasible" and due:
                self.active.add(security)
            if row + 1 >= len(self.data.dates) or self.data.dates[row + 1] > self.config.end:
                self.active.add(security)
            if self.config.exit_mode in {"strict_end", "force_cash"} and row + 1 < len(self.data.dates) and self.data.dates[row + 1] == self.config.end:
                self.active.add(security)
            if security in self.active and security not in self.started:
                self.started[security] = {"triggered_on":self.data.dates[row],
                                          "planned_start":self.data.dates[row+1] if row+1<len(self.data.dates) else None}
            estimate["exit_active"] = security in self.active
            estimate["planned_action"] = "持续退出" if security in self.active else "按策略持有"
            estimate.update(self.progress(security, position.quantity))
            self.records.append(estimate)
        for security in sorted(self.active - set(ledger.positions) - self.completed):
            if security not in self.started:
                continue
            estimate = self.estimate(security, ZERO, row)
            estimate.update(exit_active=True, planned_action="原持仓已结清", **self.progress(security, ZERO))
            self.records.append(estimate)
            self.completed.add(security)
        return self.active

    def progress(self, security, remaining):
        plan = self.started.get(security, {})
        start = plan.get("planned_start")
        trades = [trade for trade in self.broker.trades if trade["security"]==security and trade["side"]=="sell" and start and trade["date"]>=start]
        return {"triggered_on":plan.get("triggered_on"),"planned_start":start,
                "actual_start":min((trade["date"] for trade in trades),default=None),
                "sold_quantity":str(sum((dec(trade["quantity"]) for trade in trades),ZERO)),"remaining_quantity":str(remaining)}

    def allow_buy(self, security, quantity, row):
        if security in self.active:
            return False
        if self.config.exit_mode != "latest_feasible":
            return row + 1 < len(self.data.dates) and self.data.dates[row + 1] < self.config.end
        return self.actual_path.prospective(security, quantity, row)
