from decimal import Decimal

import numpy as np
import pandas as pd

from .rules import candidate_limits, tick
from .utils import dec


def detect_day(data, features, status, row):
    """将 GPU 批量检测的结果与原始精度价格边界组合为可追溯诊断。"""
    day = data.dates[row]
    valid = features.flags["valid_ohlc"][row]
    previous = features.flags["valid_ohlc"][row-1] if row else np.zeros_like(valid)
    observed = features.flags["valid_ohlc"][:row].any(axis=0) if row else np.zeros_like(valid)
    output = []

    def add(column,kind,detail,basis="矩阵自动检测"):
        output.append({"date":day,"security":data.securities[column],"type":kind,"detail":detail,"basis":basis})

    for j in np.flatnonzero(valid & ~observed):
        add(j,"first_observed","载入窗口内首次有效行情；正式上市时间由证券状态明细提供")
    for j in np.flatnonzero(~valid & previous):
        add(j,"quote_interrupted","有效行情中断；持仓延续最近有效估值")
    for j in np.flatnonzero(valid & ~previous & observed):
        add(j,"quote_resumed","有效行情恢复，持仓恢复最新估值")
    for key,label in [("zero_amount","zero_amount"),("invalid_ohlc","invalid_ohlc")]:
        for j in np.flatnonzero(features.flags[key][row]):
            add(j,label,"零成交额，开盘有效性核验限制交易" if key=="zero_amount" else "高低开收的关系异常")
    for j in np.flatnonzero(features.flags["factor_change"][row]):
        old_row,old=data.previous("cumadj",row,j)
        current=data.arrays["cumadj"][row,j]
        ratio=current/old if old else None
        add(j,"factor_change",f"{old} → {current}；比值 {ratio}；微小变化标签 {bool(ratio is not None and abs(ratio-1)<.0001)}",
            "原始精度因子变化；公司行动细项由显式明细解释")
    if row:
        close=data.arrays["close"][row-1]
        factors=data.arrays["cumadj"]
        ready=valid & previous & np.isfinite(factors[row]) & np.isfinite(factors[row-1])
        approximate=close*factors[row-1]/factors[row]
        o,h,l,c=(data.arrays[key][row] for key in ("open","high","low","close"))
        near=ready & ((h >= approximate*1.049) | (l <= approximate*.951))
        for j in np.flatnonzero(near):
            bands=candidate_limits(data.securities[j],day,close[j],factors[row-1,j],factors[row,j],
                                   bool(status.values[row,j]) if status.values[row,j]>=0 else None)
            for band in bands:
                up,down=dec(band["upper"]),dec(band["lower"])
                opening,high,low,closing=(tick(values[j]) for values in (o,h,l,c))
                if high >= up or low <= down:
                    add(j,"candidate_limit_touch",f"候选上界 {up}，下界 {down}，日内高低 {high}/{low}",band["basis"])
                for direction,boundary in (("up",up),("down",down)):
                    if opening == boundary:
                        full=high==low==boundary
                        add(j,"one_price_board" if full else "open_limit_released",
                            f"{direction}：开盘 {opening}；全天一字 {full}；收盘 {closing}",
                            "候选价格边界推断；方向性开盘成交规则独立执行")
    current_trades=data.arrays["num_trades"][row]
    unusual=(valid & ((~np.isfinite(current_trades)) | ((current_trades<=0)&(data.arrays["amt"][row]>0))))
    for j in np.flatnonzero(unusual):
        add(j,"trade_count_diagnostic","成交笔数缺失或与正成交额存在差异；辅助诊断")
    return output
