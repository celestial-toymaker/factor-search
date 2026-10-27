from __future__ import annotations

from dataclasses import dataclass, replace
from decimal import Decimal, ROUND_FLOOR, ROUND_HALF_UP

from .utils import dec, money


@dataclass(frozen=True)
class QuantityRule:
    board: str
    minimum: int
    step: int
    maximum: int


def quantity_rule(security: str, overrides=None) -> QuantityRule:
    if security.startswith(("688", "689")):
        result = QuantityRule("科创板", 200, 1, 100_000)
    elif security.startswith(("300", "301", "302")):
        result = QuantityRule("创业板", 100, 100, 300_000)
    else:
        result = QuantityRule("沪深主板", 100, 100, 1_000_000)
    result = replace(result, **{key:int(value) for key,value in (overrides or {}).items() if key in {"minimum","step","maximum"}})
    if min(result.minimum,result.step,result.maximum)<=0 or result.minimum>result.maximum:
        raise ValueError("数量规则应为正数，最低数量至多等于单笔上限")
    return result


def legal_quantity(security: str, requested, side: str, full_holding=None, rule=None) -> int:
    rule = rule or quantity_rule(security)
    quantity = max(0, int(dec(requested).to_integral_value(rounding=ROUND_FLOOR)))
    if side == "sell" and full_holding is not None:
        held = int(dec(full_holding).to_integral_value(rounding=ROUND_FLOOR))
        if quantity >= held and held <= rule.maximum:
            return held
    quantity = min(quantity, rule.maximum)
    if quantity < rule.minimum:
        return 0
    return quantity // rule.step * rule.step


def split_quantity(security: str, quantity, side: str, rule=None):
    remaining = int(dec(quantity).to_integral_value(rounding=ROUND_FLOOR))
    pieces = []
    while remaining > 0:
        piece = legal_quantity(security, remaining, side, remaining if side == "sell" else None, rule)
        if piece <= 0:
            break
        pieces.append(piece)
        remaining -= piece
    return pieces


def tick(price):
    return dec(price).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def candidate_limits(security: str, day: str, previous_close, previous_factor=1,
                     current_factor=1, is_st: bool | None = None, unlimited=False, limit_rate=None):
    if unlimited or previous_close is None or dec(previous_close) <= 0:
        return []
    factor_ratio = dec(previous_factor) / dec(current_factor)
    reference = dec(previous_close) * factor_ratio
    board = quantity_rule(security).board
    if limit_rate is not None:
        rates = [dec(limit_rate)]
    elif board == "科创板" or (board == "创业板" and day >= "2020-08-24"):
        rates = [Decimal("0.20")]
    elif day >= "2026-07-06":
        rates = [Decimal("0.10")]
    elif is_st is True:
        rates = [Decimal("0.05")]
    elif is_st is False:
        rates = [Decimal("0.10")]
    else:
        rates = [Decimal("0.05"), Decimal("0.10")]
    return [{"rate": str(rate), "reference": str(reference), "upper": str(tick(reference * (1 + rate))),
             "lower": str(max(Decimal("0.01"), tick(reference * (1 - rate)))),
             "basis": "历史规则与复权参考价候选", "st_basis": "已核状态" if is_st is not None else "保守候选"}
            for rate in rates]


def transfer_rate(security: str, day: str):
    if security.startswith("689"):
        return Decimal("0.00002")
    return Decimal("0.00001") if day >= "2022-04-29" else Decimal("0.00002")


class FeeModel:
    def __init__(self, config):
        self.config = config

    def quote(self, security, day, side, notional, prior_notional=Decimal(0), prior_commission=Decimal(0)):
        amount = money(notional)
        commission = max(money((prior_notional + amount) * dec(self.config.commission_rate)), dec(self.config.min_commission))
        commission = max(Decimal(0), commission - prior_commission) if amount > 0 else Decimal(0)
        return {
            "commission": money(commission),
            "stamp_tax": money(amount * dec(self.config.stamp_tax_rate)) if side == "sell" else Decimal(0),
            "transfer": money(amount * transfer_rate(security, day)),
            "slippage": money(amount * dec(self.config.slippage_rate)),
        }


RULE_SOURCES = [
    {"subject": "2025 年交易日历", "url": "https://www.sse.com.cn/disclosure/dealinstruc/closed/c/c_20241223_10767110.shtml"},
    {"subject": "2026 年国庆休市", "url": "https://www.sse.com.cn/disclosure/announcement/general/c/c_20260915_10832273.shtml"},
    {"subject": "2026-07-06 主板风险警示涨跌幅调整", "url": "https://www.sse.com.cn/aboutus/mediacenter/hotandd/c/c_20260424_10816474.shtml"},
    {"subject": "证券交易印花税减半", "url": "https://shanxi.chinatax.gov.cn/web/detail/sx-11400-545-1780448"},
    {"subject": "零股卖出规则", "url": "https://www.sse.com.cn/lawandrules/guide/stock/jyglywznylc/tz/c/c_20230209_5716007.shtml"},
    {"subject": "上交所股票投资费用说明与过户费", "url": "https://one.sse.com.cn/onething/gptz/"},
    {"subject": "中国结算历史收费调整记录", "url": "https://m.chinaclear.cn/zdjs/dsj/about_dsj.shtml"},
]
