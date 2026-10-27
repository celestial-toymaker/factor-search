from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, ROUND_FLOOR, getcontext
import copy

from .events import CORPORATE_TYPES, PHASES, Event
from .utils import dec, money

getcontext().prec = 38
ZERO = Decimal(0)


@dataclass
class Position:
    security: str
    quantity: Decimal = ZERO
    cost: Decimal = ZERO
    buy_fees: Decimal = ZERO
    last_price: Decimal = ZERO
    factor: Decimal = Decimal(1)
    stale_days: int = 0
    last_quote_date: str = ""
    lots: list = field(default_factory=list)
    reserved: Decimal = ZERO

    def sellable(self, day):
        locked = sum((quantity for unlock, quantity in self.lots if unlock > day), ZERO)
        return max(ZERO, self.quantity - locked - self.reserved)

    @property
    def market_value(self):
        return money(self.quantity * self.last_price)


@dataclass(frozen=True)
class Portfolio:
    cash: Decimal
    available_cash: Decimal
    frozen_cash: Decimal
    net_asset_value: Decimal
    positions: dict
    receivables: Decimal
    pending_equity: Decimal
    liabilities: Decimal


class Ledger:
    def __init__(self, initial_cash="1000000000", mode="total_return", day="2025-01-02"):
        self.mode = mode
        self.day = day
        self.phase = "initialize"
        self.cash = ZERO
        self.frozen = ZERO
        self.positions: dict[str, Position] = {}
        self.receivables: dict[str, dict] = {}
        self.pending: dict[str, dict] = {}
        self.liabilities: dict[str, Decimal] = {}
        self.entitlements = {}
        self.status_overrides = {}
        self.rules = {}
        self.processed_events = set()
        self.snapshots = {}
        self.journal = []
        self.balances = {}
        self.fees = []
        self.event_records = []
        self.handlers = {}
        self.unit_count = dec(initial_cash)
        self.last_unit_price = Decimal(1)
        self.net_external_flows = ZERO
        self.source_event = "initial-capital"
        self.post("initial_capital", {"cash:available": dec(initial_cash), "equity:capital": -dec(initial_cash)})

    @property
    def available(self):
        return self.cash - self.frozen

    @property
    def nav(self):
        return money(self.cash + sum((position.market_value for position in self.positions.values()), ZERO) +
                     sum((record["amount"] for record in self.receivables.values()), ZERO) +
                     sum((self.pending_value(record) for record in self.pending.values()), ZERO) -
                     sum(self.liabilities.values(), ZERO))

    @property
    def unit_nav(self):
        return self.nav / self.unit_count if self.unit_count > 0 else self.last_unit_price

    @staticmethod
    def pending_value(record):
        return money(record["quantity"] * record["valuation"])

    def portfolio(self):
        return Portfolio(self.cash, self.available, self.frozen, self.nav,
                         {security: {"quantity": position.quantity, "sellable": position.sellable(self.day),
                                     "market_value": position.market_value, "last_price": position.last_price,
                                     "cost": position.cost, "buy_fees": position.buy_fees}
                          for security, position in self.positions.items()},
                         sum((r["amount"] for r in self.receivables.values()), ZERO),
                         sum((self.pending_value(r) for r in self.pending.values()), ZERO),
                         sum(self.liabilities.values(), ZERO))

    def post(self, kind, entries, security=None, detail=None):
        entries = {account: money(amount) for account, amount in entries.items() if money(amount) != 0}
        if sum(entries.values(), ZERO) != 0:
            raise AssertionError(f"分录需要借贷平衡：{kind} {entries}")
        entry_id = f"J{len(self.journal) + 1:08d}"
        for account, amount in entries.items():
            self.balances[account] = self.balances.get(account, ZERO) + amount
            if account.startswith("cash:"):
                self.cash += amount
            if account == "cash:frozen":
                self.frozen += amount
            self.journal.append({"entry_id": entry_id, "sequence": len(self.journal), "date": self.day,
                                 "phase": self.phase, "type": kind, "source_event": self.source_event,
                                 "security": security, "account": account, "debit": str(max(amount, ZERO)),
                                 "credit": str(max(-amount, ZERO)), "amount": str(amount), "detail": detail or ""})
        return entry_id

    def reserve_cash(self, requested):
        amount = min(max(ZERO, money(requested)), self.available)
        self.post("cash_freeze", {"cash:frozen": amount, "cash:available": -amount})
        return amount

    def release_cash(self, amount):
        amount = min(money(amount), self.frozen)
        self.post("cash_release", {"cash:frozen": -amount, "cash:available": amount})

    def charge_fees(self, fees, security=None, order_id="", trade_id="", allow_liability=False):
        for kind, amount in fees.items():
            amount = money(amount)
            if amount == 0:
                continue
            if amount > self.available and not allow_liability:
                raise ValueError("费用需要足够可用现金")
            if amount > self.available:
                self.accrue_charge(kind, amount, security)
            else:
                self.post("fee_" + kind, {"cash:available": -amount, "expense:" + kind: amount}, security)
            self.fees.append({"date": self.day, "order_id": order_id, "trade_id": trade_id,
                              "security": security, "type": kind, "amount": str(amount)})

    def buy(self, security, quantity, price, fees, unlock_date, factor=1, order_id="", trade_id=""):
        quantity, price = dec(quantity), dec(price)
        gross = money(quantity * price)
        if gross + sum(fees.values(), ZERO) > self.available:
            raise ValueError("买入金额与费用需要足够可用现金")
        position = self.positions.setdefault(security, Position(security))
        position.quantity += quantity
        position.cost += gross
        position.buy_fees += sum(fees.values(), ZERO)
        position.last_price = price
        position.last_quote_date = self.day
        position.factor = dec(factor)
        position.lots.append((unlock_date, quantity))
        self.post("buy", {"cash:available": -gross, f"asset:cost:{security}": gross}, security)
        self.charge_fees(fees, security, order_id, trade_id)

    def sell(self, security, quantity, price, fees, order_id="", trade_id="", allow_fee_liability=False):
        position = self.positions[security]
        quantity, price = dec(quantity), dec(price)
        if quantity > position.sellable(self.day):
            raise ValueError("卖出数量应在当日可卖持仓内")
        old_quantity = position.quantity
        basis = position.cost if quantity == old_quantity else money(position.cost * quantity / old_quantity)
        buy_fees = position.buy_fees if quantity == old_quantity else money(position.buy_fees * quantity / old_quantity)
        gross = money(quantity * price)
        if not allow_fee_liability and gross + self.available < sum(fees.values(), ZERO):
            raise ValueError("卖出所得与可用现金应足以支付费用")
        position.quantity -= quantity
        position.cost -= basis
        position.buy_fees -= buy_fees
        self.post("sell", {"cash:available": gross, f"asset:cost:{security}": -basis,
                           "income:realized": basis - gross}, security)
        self.charge_fees(fees, security, order_id, trade_id, allow_fee_liability)
        tail = ZERO
        if self.mode == "total_return" and ZERO < position.quantity < 1 and position.sellable(self.day) == position.quantity:
            tail = money(position.quantity * price)
            self.post("proxy_fraction_settlement", {"cash:available": tail, f"asset:cost:{security}": -position.cost,
                                                    "income:realized": position.cost - tail}, security,
                      "总收益代理分数权益按同次成交价结算；交易费用按实际整股成交金额计算")
            basis += position.cost
            buy_fees += position.buy_fees
            position.quantity = ZERO
            position.cost = ZERO
            position.buy_fees = ZERO
        if position.quantity == 0:
            self.positions.pop(security)
        return {"gross": gross, "cost": basis, "allocated_buy_fees": buy_fees,
                "proxy_fraction_cash": tail,
                "net_realized_pnl": money(gross + tail - basis - buy_fees - sum(fees.values(), ZERO))}

    def apply_factor(self, security, factor):
        if self.mode != "total_return" or security not in self.positions:
            return None
        position = self.positions[security]
        factor = dec(factor)
        if factor <= 0 or factor == position.factor:
            return None
        previous = position.factor
        ratio = factor / previous
        before = position.quantity
        position.quantity *= ratio
        position.reserved *= ratio
        position.lots = [(day, quantity * ratio) for day, quantity in position.lots]
        position.last_price /= ratio
        position.factor = factor
        return {"before_factor": str(previous), "after_factor": str(factor), "ratio": str(ratio),
                "before_quantity": str(before), "after_quantity": str(position.quantity),
                "basis": "累计复权因子的总收益等价单位；含开盘前生效推定"}

    def unlock(self, day):
        self.day = day
        for position in self.positions.values():
            position.lots = [(unlock, quantity) for unlock, quantity in position.lots if unlock > day]
        for key, record in list(self.pending.items()):
            if record.get("kind") == "fraction" or record["tradable_at"] > day:
                continue
            security = record["security"]
            position = self.positions.setdefault(security, Position(security))
            position.quantity += record["quantity"]
            position.cost += record["cost"]
            if position.last_price <= 0:
                position.last_price = record["valuation"]
            self.source_event = key + ":delivery"
            self.post("equity_delivery", {f"asset:pending:{key}": -record["cost"],
                                          f"asset:cost:{security}": record["cost"]}, security)
            self.pending.pop(key)
        for key, record in list(self.receivables.items()):
            if record.get("auto_pay") and record.get("pay_at", "9999") <= day:
                self.source_event = f"{key}:payment:{day}"
                self.pay_receivable(key)
        self.pay_liabilities()

    def mark(self, quotes, day):
        self.day = day
        for security, position in self.positions.items():
            value = quotes.get(security)
            if value is not None and dec(value) > 0:
                position.last_price = dec(value)
                position.last_quote_date = day
                position.stale_days = 0
            else:
                position.stale_days += 1
        for record in self.pending.values():
            if record["security"] in quotes:
                record["valuation"] = dec(quotes[record["security"]])
        self.snapshots[day] = {security: position.quantity for security, position in self.positions.items()}

    def pay_receivable(self, key):
        record = self.receivables.pop(key, None)
        if record:
            self.post("receivable_paid", {"cash:available": record["amount"], f"asset:receivable:{key}": -record["amount"]},
                      record.get("security"))

    def pay_liabilities(self):
        for key, due in list(self.liabilities.items()):
            paid = min(due, self.available)
            if paid > 0:
                self.post("liability_paid", {"cash:available": -paid, "liability:" + key: paid})
                self.liabilities[key] -= paid
            if self.liabilities[key] == 0:
                self.liabilities.pop(key)

    def accrue_charge(self, kind, amount, security=None):
        amount = money(amount)
        if amount < 0:
            raise ValueError("税费应计金额应为零或正数")
        self.liabilities[kind] = self.liabilities.get(kind, ZERO) + amount
        self.post(kind + "_accrual", {"expense:" + kind: amount, "liability:" + kind: -amount}, security)
        self.pay_liabilities()

    def apply(self, event: Event):
        if event.event_id in self.processed_events:
            return {"duplicate": True}
        kind, payload, security = event.type, event.payload, event.security
        if self.mode == "total_return" and kind in CORPORATE_TYPES:
            raise ValueError("请为显式公司行动选择 explicit 口径，以使经济影响计入一次")
        self.source_event = event.event_id
        before = self.nav
        action_id = payload.get("action_id", event.event_id)
        position = self.positions.get(security)
        quantity = position.quantity if position else ZERO
        if kind in self.handlers:
            self.handlers[kind](self, event)
        elif kind in {"cash_deposit", "cash_withdrawal"}:
            amount = money(payload["amount"])
            if amount <= 0:
                raise ValueError("资金流金额应大于零")
            if kind == "cash_withdrawal":
                if amount > self.available:
                    raise ValueError("出金金额应在可用现金范围内")
                amount = -amount
            unit_price = self.unit_nav
            self.last_unit_price = unit_price
            self.unit_count += amount / unit_price
            if abs(self.unit_count) < Decimal("1e-25"):
                self.unit_count = ZERO
            self.net_external_flows += amount
            self.post(kind, {"cash:available": amount, "equity:capital": -amount})
        elif kind == "dividend_record":
            if dec(payload["cash_per_share"]) < 0:
                raise ValueError("每股现金分红应为零或正数")
            record_date = payload.get("record_date", self.day)
            entitled = self.snapshots.get(record_date, {}).get(security, quantity if record_date == self.day else ZERO)
            amount = money(entitled * dec(payload["cash_per_share"]))
            self.entitlements[action_id] = {"quantity": entitled, "amount": amount, "security": security}
        elif kind == "dividend_ex":
            entitlement = self.entitlements.get(action_id)
            if entitlement is None:
                record_date = payload.get("record_date")
                if not record_date or record_date not in self.snapshots:
                    raise ValueError("分红除息需要登记日持仓快照")
                entitled = self.snapshots[record_date].get(security, ZERO)
                entitlement = {"quantity": entitled, "amount": money(entitled * dec(payload["cash_per_share"])), "security": security}
                self.entitlements[action_id] = entitlement
            amount = entitlement["amount"]
            self.receivables[action_id] = {"amount": amount, "security": security}
            if position and position.last_quote_date < event.effective_at.date().isoformat():
                per_share = dec(payload.get("cash_per_share", amount / entitlement["quantity"] if entitlement["quantity"] else 0))
                position.last_price = max(ZERO, position.last_price - per_share)
            self.post(kind, {f"asset:receivable:{action_id}": amount, "income:dividend": -amount}, security)
        elif kind == "dividend_pay":
            self.pay_receivable(action_id)
        elif kind == "split":
            ratio = dec(payload["ratio"])
            if ratio <= 0:
                raise ValueError("拆并股比例应大于零")
            if position:
                total = position.quantity * ratio
                whole = total.to_integral_value(rounding=ROUND_FLOOR)
                fraction = total - whole
                fraction_cost = money(position.cost * fraction / total) if total else ZERO
                position.quantity = whole
                position.cost -= fraction_cost
                if position.last_quote_date < event.effective_at.date().isoformat():
                    position.last_price /= ratio
                position.lots = [(day, shares * ratio) for day, shares in position.lots]
                position.reserved *= ratio
                if fraction:
                    key = action_id + ":fraction"
                    self.pending[key] = {"security": security, "quantity": fraction, "cost": fraction_cost,
                                         "valuation": position.last_price, "tradable_at": "9999-12-31", "kind": "fraction"}
                    self.post("fraction_entitlement", {f"asset:cost:{security}": -fraction_cost,
                                                       f"asset:pending:{key}": fraction_cost}, security)
                if position.quantity == 0:
                    self.positions.pop(security)
        elif kind in {"stock_dividend", "rights_offer", "rights_subscribe"}:
            if kind == "rights_offer":
                if dec(payload["ratio"]) < 0:
                    raise ValueError("配股权益比例应为零或正数")
                maximum = (quantity * dec(payload["ratio"])).to_integral_value(rounding=ROUND_FLOOR)
                self.entitlements[action_id] = {"quantity": maximum, "security": security, "disposition": "默认放弃"}
            elif kind == "rights_subscribe":
                entitlement = self.entitlements.get(action_id)
                if entitlement is None:
                    raise ValueError("配股申购需要对应权益登记")
                shares = dec(payload["quantity"])
                cost = money(shares * dec(payload["subscription_price"]))
                if shares <= 0 or shares != int(shares) or shares > entitlement["quantity"] or cost < 0 or cost > self.available:
                    raise ValueError("请核对配股可申购股数与可用资金")
                key = event.event_id
                self.pending[key] = {"security": security, "quantity": shares, "cost": cost,
                                     "valuation": dec(payload.get("valuation", payload["subscription_price"])),
                                     "tradable_at": payload["tradable_at"][:10], "kind": "rights"}
                entitlement["quantity"] -= shares
                entitlement["disposition"] = "已申购"
                self.post(kind, {"cash:available": -cost, f"asset:pending:{key}": cost}, security)
            elif position:
                shares = quantity * dec(payload["ratio"])
                if shares < 0:
                    raise ValueError("送转比例应为零或正数")
                cost = money(position.cost * shares / (quantity + shares)) if quantity + shares else ZERO
                position.cost -= cost
                if position.last_quote_date < event.effective_at.date().isoformat():
                    position.last_price = position.last_price * quantity / (quantity + shares)
                whole = shares.to_integral_value(rounding=ROUND_FLOOR)
                fraction = shares - whole
                allocated_total = ZERO
                for suffix, count, category in [("shares", whole, "bonus"), ("fraction", fraction, "fraction")]:
                    if count:
                        key = action_id + ":" + suffix
                        allocated = cost - allocated_total if suffix == "fraction" or fraction == 0 else money(cost * count / shares)
                        allocated_total += allocated
                        self.pending[key] = {"security": security, "quantity": count, "cost": allocated,
                                             "valuation": position.last_price, "tradable_at": payload["tradable_at"][:10], "kind": category}
                        self.post(kind, {f"asset:cost:{security}": -allocated, f"asset:pending:{key}": allocated}, security)
        elif kind in {"conversion", "merger", "spinoff", "code_change"}:
            target = payload["target_security"]
            if position:
                ratio = dec(payload.get("ratio", 1))
                if ratio <= 0:
                    raise ValueError("转换比例应大于零")
                if kind == "code_change":
                    if target in self.positions:
                        raise ValueError("代码转换目标已有持仓，请使用转换事件合并权益")
                    self.positions[target] = self.positions.pop(security)
                    self.positions[target].security = target
                    self.post(kind, {f"asset:cost:{security}": -position.cost, f"asset:cost:{target}": position.cost}, target)
                else:
                    allocation = dec(payload["cost_allocation"]) if kind == "spinoff" else Decimal(1)
                    if not 0 <= allocation <= 1:
                        raise ValueError("成本分配比例应在 0 至 1 之间")
                    cost = money(position.cost * allocation)
                    shares = quantity * ratio
                    cash = money(quantity * dec(payload.get("cash_per_share", 0)))
                    default_value = max(ZERO, position.market_value - cash) / shares
                    valuation = dec(payload.get("valuation", default_value))
                    if kind == "spinoff" and "valuation" not in payload:
                        raise ValueError("分拆需要新增权益的估值依据")
                    whole = shares.to_integral_value(rounding=ROUND_FLOOR)
                    fraction = shares - whole
                    allocated_total = ZERO
                    for suffix, count, category in [("shares", whole, kind), ("fraction", fraction, "fraction")]:
                        if count:
                            key = action_id + ":" + suffix
                            allocated = cost - allocated_total if suffix == "fraction" or fraction == 0 else money(cost * count / shares)
                            allocated_total += allocated
                            self.pending[key] = {"security": target, "quantity": count, "cost": allocated,
                                                 "valuation": valuation, "tradable_at": payload["tradable_at"][:10], "kind": category}
                            self.post(kind, {f"asset:cost:{security}": -allocated, f"asset:pending:{key}": allocated}, security)
                    position.cost -= cost
                    if kind == "spinoff" and position.last_quote_date < event.effective_at.date().isoformat():
                        position.last_price = max(ZERO, (position.market_value - shares * valuation - cash) / quantity)
                    elif kind != "spinoff":
                        self.positions.pop(security)
                    if cash:
                        cash_key = action_id + ":cash"
                        self.receivables[cash_key] = {"amount": cash, "security": security, "auto_pay": True,
                                                      "pay_at": payload["pay_at"][:10]}
                        self.post("conversion_cash", {f"asset:receivable:{cash_key}": cash, "income:corporate_cash": -cash}, security)
        elif kind == "delist_settlement":
            if dec(payload["price"]) < 0:
                raise ValueError("退市现金对价应为零或正数")
            if position:
                amount = money(quantity * dec(payload["price"]))
                self.receivables[action_id] = {"amount": amount, "security": security, "auto_pay": True,
                                               "pay_at": payload["pay_at"][:10]}
                self.post(kind, {f"asset:cost:{security}": -position.cost, f"asset:receivable:{action_id}": amount,
                                 "income:realized": position.cost - amount}, security)
                self.positions.pop(security)
            self.status_overrides[security] = {"delisted": True}
        elif kind == "fraction_settlement":
            record = self.pending.pop(action_id)
            amount = money(payload["amount"]) if "amount" in payload else money(record["quantity"] * dec(payload["price"]))
            self.post(kind, {f"asset:pending:{action_id}": -record["cost"], "cash:available": amount,
                             "income:realized": record["cost"] - amount}, record["security"])
        elif kind in {"tax", "fee"}:
            self.accrue_charge(kind, payload["amount"], security)
            self.fees.append({"date":self.day,"order_id":"","trade_id":"","security":security,
                              "type":kind,"amount":str(money(payload["amount"])),"event_id":event.event_id,
                              "source":event.source,"basis":"显式金额应计；已付现金与应付负债分别记账"})
        elif kind in {"security_status", "delist_warning", "listing"}:
            if "is_st" in payload:
                from .status import StatusProvider
                payload = {**payload,"is_st":StatusProvider.parse_flag(payload["is_st"])}
            self.status_overrides.setdefault(security, {}).update(payload)
            if kind == "delist_warning":
                self.status_overrides[security]["delist_warning"] = True
        elif kind == "rule_change":
            self.rules.setdefault(security or "market", {}).update(payload)
        elif kind not in {"order_cancel", "order_replace", "user"}:
            raise ValueError("请登记事件处理器：" + kind)
        self.processed_events.add(event.event_id)
        impact = {"event_id": event.event_id, "type": kind, "security": security, "date": self.day,
                  "effective_at": event.effective_at.isoformat(), "known_at": event.known_at.isoformat(),
                  "applied_at": self.day + "T" + PHASES.get(self.phase,"15:05:00") + "+08:00",
                  "nav_before": str(before), "nav_after": str(self.nav), "cash_after": str(self.cash), "source": event.source}
        self.event_records.append(impact)
        return impact

    def register_handler(self, event_type, handler):
        self.handlers[event_type] = handler

    def assert_invariants(self):
        assert self.cash >= 0, "现金应为零或正数"
        assert ZERO <= self.frozen <= self.cash, "冻结现金应在现金总额内"
        assert self.balances.get("cash:available", ZERO) == self.available
        assert self.balances.get("cash:frozen", ZERO) == self.frozen
        assert sum(self.balances.values(), ZERO) == 0, "总账应借贷平衡"
        for security, position in self.positions.items():
            assert position.quantity >= 0 and position.reserved >= 0
            assert self.balances.get(f"asset:cost:{security}", ZERO) == position.cost, "持仓成本应与总账一致"

    def position_rows(self):
        return [{"date": self.day, "security": security, "quantity": str(p.quantity), "cost": str(p.cost),
                 "cost_including_buy_fees": str(p.cost + p.buy_fees), "market_value": str(p.market_value),
                 "last_price": str(p.last_price), "last_quote_date": p.last_quote_date, "stale_days": p.stale_days,
                 "sellable": str(p.sellable(self.day)), "reserved": str(p.reserved), "factor": str(p.factor),
                 "unit_basis": "总收益等价股数" if self.mode == "total_return" else "原始股数"}
                for security, p in sorted(self.positions.items())]
