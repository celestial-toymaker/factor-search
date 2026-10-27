"""A 股全市场回测与研究工作台。"""

__version__ = "0.1.0"


def run(config=None, strategy=None, **kwargs):
    """按相同配置运行公共引擎，返回 BacktestResult。"""
    from .engine import run as execute
    return execute(config, strategy, **kwargs)


def __getattr__(name):
    from importlib import import_module
    modules = {"Config":"config", "Strategy":"strategies", "TargetWeights":"strategies", "StrategyRegistry":"strategies",
               "ComposedStrategy":"strategies", "Event":"events", "EventSource":"events", "FactorProvider":"data", "BacktestResult":"engine"}
    if name in modules:
        value = getattr(import_module(f".{modules[name]}", __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(name)

