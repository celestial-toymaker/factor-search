from __future__ import annotations

from dataclasses import asdict, dataclass, field
from decimal import Decimal
from importlib.metadata import entry_points
import importlib

import numpy as np

from .utils import dec


@dataclass
class TargetWeights:
    weights: dict[str, Decimal]
    reason: str = "策略目标"
    rebalance: bool = True

    def __post_init__(self):
        self.weights = {security: dec(weight) for security, weight in self.weights.items()}
        if any(weight < 0 for weight in self.weights.values()) or sum(self.weights.values(), Decimal(0)) > 1:
            raise ValueError("现货多头目标权重应为零或正数，合计至多为 1")


@dataclass(frozen=True)
class StrategySpec:
    id: str
    name: str
    family: str
    description: str
    example: str
    risk: str
    defaults: dict = field(default_factory=dict)
    required_fields: tuple = ("open", "high", "low", "close", "amt", "cumadj", "is_st")
    min_history: int = 60
    stateful: bool = False

    def public(self):
        return asdict(self)


SPECS = [
    StrategySpec("momentum", "动量轮动", "顺势", "从沪深非 ST 全市场里，选择过去一段时间涨幅靠前的股票。",
                 "甲从 10 元涨到 12 元，乙涨到 11 元；相同窗口下，甲的动量排名更高。", "上涨趋势反转时，排名信号会滞后。", {"window": 20}),
    StrategySpec("liquidity", "高流动性等权", "基准", "选择成交额较高的股票，将资金平均分配。",
                 "选中 20 只股票时，95% 目标资金平均分成 20 份。", "成交活跃体现交易便利，仍需承担价格波动。"),
    StrategySpec("buy_hold", "单标的买入持有", "入门", "买入一只指定的非 ST 股票，持有至计划退出。",
                 "以 10 元买入 100 股，股价到 11 元时，股票市值增加 100 元，再考虑交易成本。", "单一股票的风险集中。", {"symbol": "000001.SZ"}),
    StrategySpec("reversal", "短期反转", "回归", "从全市场选择近期跌幅较大的股票，观察随后回升。",
                 "甲近 5 日跌 8%，乙跌 2%；甲的反转候选排名更高。", "短期下跌可能继续延续。", {"window": 5}),
    StrategySpec("moving_average", "双均线趋势", "顺势", "短期平均价格高于长期平均价格时持有，反向时退出。",
                 "20 日均价 11 元、60 日均价 10 元时，趋势条件成立。", "震荡行情可能带来反复交易。", {"fast": 20, "slow": 60}, stateful=True),
    StrategySpec("breakout", "通道突破", "顺势", "收盘突破此前高点进入，跌破此前低点退出。",
                 "此前 20 日最高 10 元，今天收在 10.20 元，形成向上突破。", "突破后回落会产生假信号。", {"window": 20, "exit_window": 10}, stateful=True),
    StrategySpec("low_volatility", "低波动选股", "稳健因子", "比较全市场近期收益波动，选择波动较小的股票。",
                 "甲每天变化约 1%，乙约 4%；相同窗口下，甲的历史波动更低。", "较低的历史波动也可能伴随之后的大幅变动。", {"window": 20}),
    StrategySpec("rsi", "RSI 均值回归", "回归", "近期下跌力量较强时进入，力量恢复平衡时退出。",
                 "14 日 RSI 跌到 28 时满足进入条件，回升到 50 时退出。", "持续下跌期间可能较早进入。", {"window": 14, "entry": 30, "exit": 50}, stateful=True),
    StrategySpec("bollinger", "布林带回归", "回归", "价格低于近期均价的下方区间时进入，回到均价时退出。",
                 "20 日均价 10 元、标准差 0.5 元，2 倍标准差下轨为 9 元。", "价格区间会变化，价格可能沿下轨继续下行。", {"window": 20, "deviations": 2}, stateful=True),
    StrategySpec("multifactor", "多因子组合", "组合", "综合动量、低波动和短期反转的全市场排名。",
                 "一只股票三项排名分别为前 10%、20%、30%，将三项排名平均后比较。", "多个因子之间可能相互抵消，也会同时失效。", {"window": 20}),
]


class Strategy:
    def on_close(self, history, portfolio) -> TargetWeights:
        raise NotImplementedError

    def on_event(self, event, context):
        return None


class ComposedStrategy(Strategy):
    """把证券池、选股、择时、分配和调仓周期组合成一项策略。"""
    def __init__(self, config, selector, timing=lambda history, portfolio: True,
                 universe=lambda history: history.eligible, allocator=None, interval=None):
        self.config, self.selector, self.timing, self.universe = config, selector, timing, universe
        self.allocator = allocator
        self.interval = interval or config.rebalance_days
        self.calls = 0

    def on_close(self, history, portfolio):
        due = self.calls % self.interval == 0
        self.calls += 1
        if not due:
            return TargetWeights({}, "等待组合调仓周期", False)
        if not self.timing(history, portfolio):
            return TargetWeights({}, "择时条件触发现金持有")
        candidates = np.asarray(self.universe(history), dtype=bool) & history.eligible
        scores = np.asarray(self.selector(history), dtype=float)
        selected = stable_rank(scores, candidates, history.securities)[:self.config.max_positions]
        if self.allocator:
            weights = self.allocator(history, selected, portfolio)
        else:
            weights = BuiltinStrategy(self.config, SPECS[0]).allocate(history, selected.tolist())
        return TargetWeights(weights, "组合接口：证券池、选股、择时与仓位分配")


def stable_rank(scores, eligible, securities):
    indices = np.flatnonzero(eligible & np.isfinite(scores))
    return indices[np.lexsort((np.asarray(securities)[indices], -scores[indices]))]


def percentiles(scores, eligible, securities):
    indices = stable_rank(scores, eligible, securities)
    result = np.full(len(scores), np.nan)
    if len(indices):
        result[indices] = 1 - np.arange(len(indices)) / max(1, len(indices) - 1)
    return result


class BuiltinStrategy(Strategy):
    def __init__(self, config, spec):
        self.config, self.spec = config, spec
        self.parameters = {**spec.defaults, **config.parameters}
        self.calls = 0
        self.bought_once = False
        self.initial_basket = None
        self.last_scores = None
        self.last_score_date = None

    def score(self, history):
        kind, p = self.spec.id, self.parameters
        window = int(p.get("window", 20))
        if kind == "momentum":
            return history.factor(f"mom_{window}").astype(float)
        if kind == "reversal":
            return -history.factor(f"mom_{window}").astype(float)
        if kind == "liquidity":
            return history.factor("adv_20").astype(float)
        if kind == "low_volatility":
            return -history.factor(f"vol_{window}").astype(float)
        if kind == "moving_average":
            return history.factor(f"ma_{int(p['fast'])}") / history.factor(f"ma_{int(p['slow'])}") - 1
        if kind == "breakout":
            current = history.prices(periods=1, adjusted=True)[0]
            return current / history.factor(f"high_prev_{window}") - 1
        if kind == "rsi":
            return 100 - history.factor(f"rsi_{window}")
        if kind == "bollinger":
            current = history.prices(periods=1, adjusted=True)[0]
            sigma = history.factor(f"std_{window}")
            return np.divide(history.factor(f"ma_{window}") - current, sigma,
                             out=np.full(len(current), np.nan), where=sigma > 0)
        if kind == "multifactor":
            parts = [history.factor(f"mom_{window}"), -history.factor("vol_20"), -history.factor("mom_5")]
            refined = []
            for part,component,length in zip(parts,("momentum","low_volatility","reversal"),(window,20,5)):
                part = part.astype(float)
                ranked = stable_rank(part,history.eligible,history.securities)
                if len(ranked)>1:
                    tolerance = np.maximum(1e-5,np.abs(part[ranked[:-1]])*1e-5)
                    close = np.flatnonzero(np.abs(np.diff(part[ranked])) <= tolerance)
                    candidates = np.unique(np.r_[ranked[close],ranked[close+1]])
                    if len(candidates):
                        part[candidates] = self.raw_scores(history,candidates,component,length)
                refined.append(percentiles(part,history.eligible,history.securities))
            return sum(refined) / len(refined)
        return history.factor("adv_20").astype(float)

    @staticmethod
    def raw_scores(history, candidates, kind, window):
        if kind == "liquidity":
            return history.prices("amt",periods=20)[:,candidates].mean(axis=0)
        prices = history.prices(periods=window+1,adjusted=True)[:,candidates]
        if kind == "low_volatility":
            return -np.std(prices[1:]/prices[:-1]-1,axis=0)
        momentum = prices[-1]/prices[0]-1
        return -momentum if kind == "reversal" else momentum

    def refine_ranking(self, history, scores, ranked):
        """对入选边界的小集合使用原始精度复核。"""
        if len(ranked) <= self.config.max_positions or self.spec.id not in {"momentum", "reversal", "low_volatility", "liquidity"}:
            return ranked
        window = int(self.parameters.get("window", 20))
        boundary = scores[ranked[self.config.max_positions - 1]]
        candidates = ranked[scores[ranked] >= boundary - max(1e-5, abs(float(boundary)) * 1e-5)]
        scores[candidates] = self.raw_scores(history,candidates,self.spec.id,window)
        return stable_rank(scores, history.eligible, history.securities)

    def on_close(self, history, portfolio):
        day_number = self.calls
        self.calls += 1
        if self.spec.id == "buy_hold" and self.parameters.get("symbol","000001.SZ") in portfolio.positions:
            self.bought_once = True
        if self.spec.id == "buy_hold" and self.bought_once:
            return TargetWeights({}, "持有首批证券", False)
        if self.spec.id != "buy_hold" and not self.spec.stateful and day_number % self.config.rebalance_days:
            return TargetWeights({}, "等待调仓日", False)
        scores = np.asarray(self.score(history),dtype=float)
        ranked = stable_rank(scores, history.eligible, history.securities)
        ranked = self.refine_ranking(history, scores, ranked)
        if self.spec.id == "buy_hold":
            symbol = self.parameters.get("symbol", "000001.SZ")
            selected = [history.securities.index(symbol)] if symbol in history.securities and history.eligible[history.securities.index(symbol)] else []
        elif self.spec.stateful:
            selected = self.state_selection(history, portfolio, scores)
        else:
            selected = ranked[:self.config.max_positions].tolist()
        self.last_scores, self.last_score_date = scores.copy(), history.date
        weights = self.allocate(history, selected)
        if self.initial_basket is None and weights:
            self.initial_basket = dict(weights)
        return TargetWeights(weights, self.spec.name)

    def state_selection(self, history, portfolio, scores):
        p, kind = self.parameters, self.spec.id
        current = history.prices(periods=1, adjusted=True)[0]
        window = int(p.get("window", 20))
        if kind == "moving_average":
            enter, remain = scores > 0, scores > 0
        elif kind == "breakout":
            enter = scores > 0
            remain = current >= history.factor(f"low_prev_{int(p['exit_window'])}")
        elif kind == "rsi":
            rsi = 100 - scores
            enter, remain = rsi < float(p["entry"]), rsi < float(p["exit"])
        else:
            average = history.factor(f"ma_{window}")
            sigma = history.factor(f"std_{window}")
            enter = current < average - float(p["deviations"]) * sigma
            remain = current < average
        # GPU 先批量计算信号，阈值附近使用原始精度确认进入和持有条件。
        if kind in {"moving_average","breakout"}:
            boundary = np.abs(scores) <= 1e-5
            if kind == "breakout":
                low = history.factor(f"low_prev_{int(p['exit_window'])}")
                boundary |= np.abs(current-low) <= np.maximum(1e-5,np.abs(current)*2e-6)
        elif kind == "rsi":
            boundary = (np.abs(rsi-float(p["entry"]))<=.001) | (np.abs(rsi-float(p["exit"]))<=.001)
        else:
            tolerance = np.maximum(1e-5,np.abs(current)*2e-6)
            boundary = (np.abs(current-average)<=tolerance) | (np.abs(current-average+float(p["deviations"])*sigma)<=tolerance)
        candidates = np.flatnonzero(history.eligible & boundary)
        if len(candidates):
            if kind == "moving_average":
                prices = history.prices(periods=int(p["slow"]),adjusted=True)[:,candidates]
                precise = prices[-int(p["fast"]):].mean(axis=0)/prices.mean(axis=0)-1
                enter[candidates],remain[candidates] = precise>0,precise>0
                scores[candidates] = precise
            elif kind == "breakout":
                high = history.prices("high",periods=window+1,adjusted=True)[:-1,candidates].max(axis=0)
                low = history.prices("low",periods=int(p["exit_window"])+1,adjusted=True)[:-1,candidates].min(axis=0)
                enter[candidates],remain[candidates] = current[candidates]>high,current[candidates]>=low
                scores[candidates] = current[candidates]/high-1
            elif kind == "rsi":
                prices = history.prices(adjusted=True)[:,candidates]
                changes = np.diff(prices,axis=0)
                gain,loss = np.zeros(len(candidates)),np.zeros(len(candidates))
                count = np.zeros(len(candidates),dtype=int)
                for row,change in enumerate(changes):
                    valid = np.isfinite(change)
                    count = np.where(valid,count+1,0)
                    gain = np.where(valid,(gain*(window-1)+np.maximum(change,0))/window,0)
                    loss = np.where(valid,(loss*(window-1)+np.maximum(-change,0))/window,0)
                    seed = count==window
                    if seed.any():
                        recent = changes[row-window+1:row+1]
                        gain[seed] = np.maximum(recent[:,seed],0).mean(axis=0)
                        loss[seed] = np.maximum(-recent[:,seed],0).mean(axis=0)
                ratio = np.divide(gain,loss,out=np.zeros_like(gain),where=loss>0)
                precise = np.where(loss>0,100-100/(1+ratio),np.where(gain>0,100,50))
                precise = np.where(count>=window,precise,np.nan)
                enter[candidates],remain[candidates] = precise<float(p["entry"]),precise<float(p["exit"])
                scores[candidates] = 100-precise
            else:
                prices = history.prices(periods=window,adjusted=True)[:,candidates]
                average,sigma = prices.mean(axis=0),prices.std(axis=0)
                enter[candidates] = current[candidates]<average-float(p["deviations"])*sigma
                remain[candidates] = current[candidates]<average
                scores[candidates] = np.divide(average-current[candidates],sigma,out=np.full(len(candidates),np.nan),where=sigma>0)
        held = [i for i, security in enumerate(history.securities)
                if security in portfolio.positions and history.eligible[i] and remain[i]]
        held_set = set(held)
        new = history.eligible & enter & np.isfinite(scores)
        new[list(held_set)] = False
        liquid = stable_rank(history.prices("amt",periods=20).mean(axis=0), new, history.securities)
        return (held + liquid.tolist())[:self.config.max_positions]

    def allocate(self, history, indices):
        if not indices:
            return {}
        if self.parameters.get("allocation", "equal") == "inverse_volatility":
            vol = history.factor("vol_20")[indices]
            raw = np.divide(1., np.maximum(vol, 1e-6))
        else:
            raw = np.ones(len(indices))
        fractions = raw / raw.sum()
        exposure = dec(self.config.target_exposure)
        weights = {history.securities[index]: (exposure * dec(float(weight))).quantize(Decimal("0.000000000001"))
                   for index, weight in zip(indices, fractions)}
        difference = exposure - sum(weights.values(), Decimal(0))
        weights[history.securities[indices[0]]] += difference
        return weights


class StrategyRegistry:
    def __init__(self):
        self.specs = {spec.id: spec for spec in SPECS}
        self.factories = {spec.id: (lambda config, spec=spec: BuiltinStrategy(config, spec)) for spec in SPECS}
        for entry in entry_points(group="ashare_bt.strategies"):
            factory = entry.load()
            self.register(factory.spec, factory)

    def register(self, spec, factory):
        if spec.id in self.specs:
            raise ValueError("策略编号已登记：" + spec.id)
        self.specs[spec.id], self.factories[spec.id] = spec, factory

    def create(self, config):
        if config.strategy.startswith("classic_"):
            from .classic_factors import ClassicFactorStrategy
            return ClassicFactorStrategy(config)
        if config.strategy.startswith("python:"):
            module, name = config.strategy[7:].rsplit(":", 1)
            return getattr(importlib.import_module(module), name)(config)
        if config.strategy not in self.factories:
            raise ValueError("请选择已登记的策略：" + config.strategy)
        spec = self.specs[config.strategy]
        config.parameters = {**spec.defaults, **config.parameters}
        windows = [int(config.parameters[key]) for key in ("window","fast","slow","exit_window","rsi_window") if key in config.parameters]
        required = max(windows,default=20) + int(spec.id in {"momentum","reversal","low_volatility","rsi","breakout","multifactor"})
        config.min_history = max(config.min_history,spec.min_history,required)
        return self.factories[config.strategy](config)

    def list(self):
        return [spec.public() for spec in self.specs.values()]
