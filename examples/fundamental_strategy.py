"""用时点财务因子扩展选股；准备自己的带来源和公告时间的因子文件。"""
import numpy as np
import pandas as pd

from ashare_bt.data import FactorProvider
from ashare_bt.strategies import ComposedStrategy, StrategySpec


FIELD_CONTRACT = {
    "book_to_price": {"family":"价值", "unit":"比例", "fields":["归母净资产","总市值","报告期","公告及修订时间"]},
    "roe": {"family":"质量", "unit":"比例", "fields":["归母净利润","平均归母净资产","报告期","公告及修订时间"]},
    "revenue_growth": {"family":"成长", "unit":"同比比例", "fields":["本期营业收入","可比上年营业收入","报告期","公告及修订时间"]},
}


class FundamentalStrategy(ComposedStrategy):
    spec = StrategySpec("fundamental_example", "价值质量成长示例", "扩展", "按已公布财务指标选股。",
                        "ROE 为 15% 的含义是每 100 元平均净资产对应 15 元利润。", "指标解释依赖会计口径及公告时点。",
                        required_fields=("book_to_price","roe","revenue_growth","known_at","effective_at","source"))

    def __init__(self,config):
        self.provider=FactorProvider(pd.read_parquet(config.root/"data/fundamentals.parquet"))
        def selector(history):
            known=self.provider.as_of(history.date+"T15:02:00+08:00")
            table=known.pivot(index="security",columns="factor",values="value").reindex(history.securities)
            ranks=table[list(FIELD_CONTRACT)].rank(pct=True)
            return ranks.mean(axis=1,skipna=False).to_numpy(float)
        super().__init__(config,selector)
