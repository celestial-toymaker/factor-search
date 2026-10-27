from dataclasses import dataclass
from decimal import Decimal
from typing import Iterable, Protocol

from .events import Event


class ExternalEventAdapter(Protocol):
    """适配器保留原始来源与时间，将外部消息转换成幂等事件。"""
    def normalize(self, raw: dict) -> Iterable[Event]: ...


class CorporateActionHandler(Protocol):
    def __call__(self, ledger, event: Event) -> dict: ...


class AssetModel(Protocol):
    """多资产扩展需要在估值、抵押品和交割模型上同时声明口径。"""
    def market_value(self, quantity: Decimal, price: Decimal) -> Decimal: ...
    def margin(self, quantity: Decimal, price: Decimal) -> Decimal: ...
    def settlement_events(self, date: str, position: dict) -> Iterable[Event]: ...


@dataclass(frozen=True)
class DataRequirement:
    family: str
    trigger: str
    fields: tuple[str, ...]
    interface: str
    asset_model: str


EXTERNAL_REQUIREMENTS = [
    DataRequirement("tick", "交易所逐笔成交或报价时间戳推进", ("exchange_time","known_at","last","size","sequence"),
                    "ExternalEventAdapter → EventSource → Strategy.on_event", "现货标的、逐笔估值与日内时钟"),
    DataRequirement("orderbook", "盘口深度或委托队列变化", ("bid_levels","ask_levels","queue_id","order_sequence","cancel_sequence"),
                    "ExternalEventAdapter + 队列撮合器", "价格优先、时间优先、排队剩余量和撤单"),
    DataRequirement("margin", "融资负债、担保比例或借券可用量变化", ("collateral","debt","maintenance_ratio","borrowable","interest_rate","recall_date"),
                    "AssetModel.margin + EventSource", "抵押品、保证金、利息、强平与借券归还"),
    DataRequirement("futures", "合约切换、逐日盯市、最后交易日与交割日", ("contract_multiplier","settlement_price","margin_rate","expiry","delivery_terms"),
                    "AssetModel.settlement_events + EventSource", "合约乘数、保证金、每日结算与交割"),
    DataRequirement("options", "行权、到期或被指派", ("strike","expiry","right","exercise_style","multiplier","assignment","underlying"),
                    "AssetModel.settlement_events + ExternalEventAdapter", "权利金、现金或实物行权、标的交收及保证金"),
    DataRequirement("live", "券商断开、重连、外部成交或余额变动", ("connection_id","message_sequence","broker_order_id","balance_snapshot","exchange_time"),
                    "ExternalEventAdapter + 持久化检查点与对账", "实盘订单确认、幂等补报、账户差异与券商资产模型"),
    DataRequirement("fundamentals", "财报或修订数据可获知", ("security","factor","value","effective_at","known_at","source","unit","report_period"),
                    "FactorProvider.as_of + StrategyRegistry", "现货估值口径保持一致；财务指标逐期留存修订版本"),
]
