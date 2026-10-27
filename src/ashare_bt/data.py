from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import hashlib

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.feather as feather

from .config import Config
from .utils import sha256

METRICS = ("open", "high", "low", "close", "amt", "num_trades", "cumadj")


@dataclass
class MarketData:
    dates: list[str]
    securities: list[str]
    arrays: dict[str, np.ndarray]
    fingerprint: str = "synthetic"
    source_files: list[dict] = field(default_factory=list)
    arrow_nulls: dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        self.index = {security: i for i, security in enumerate(self.securities)}
        self.date_index = {day: i for i, day in enumerate(self.dates)}
        self.stock_mask = np.array([not s.startswith("689") for s in self.securities])
        expected = (len(self.dates), len(self.securities))
        if any(array.shape != expected for array in self.arrays.values()):
            raise ValueError("七个字段的日期和证券维度应保持一致")

    @property
    def shape(self):
        return len(self.dates), len(self.securities)

    def trading_indices(self, config: Config):
        return [i for i, day in enumerate(self.dates) if config.start <= day <= config.end]

    def previous(self, metric: str, row: int, column: int):
        values = self.arrays[metric][:row, column]
        positions = np.flatnonzero(np.isfinite(values) & (values > 0))
        return (int(positions[-1]), float(values[positions[-1]])) if len(positions) else (None, None)


def load_data(config: Config, progress=lambda **kw: None) -> MarketData:
    arrays, files, nulls = {}, [], {}
    dates = securities = None
    digest = hashlib.sha256()
    for number, metric in enumerate(METRICS):
        path = config.path("data_dir") / f"{metric}.feather"
        table = feather.read_table(path, memory_map=True)
        if not table.column_names or table.column_names[0] != "date":
            raise ValueError(f"{metric} 的第一列应为 date")
        dtype = table.schema.field("date").type
        lower = pa.scalar(datetime.fromisoformat(config.warmup_start), type=dtype)
        upper = pa.scalar(datetime.fromisoformat(config.end), type=dtype)
        table = table.filter(pc.and_(pc.greater_equal(table["date"], lower), pc.less_equal(table["date"], upper)))
        current_dates = [value.strftime("%Y-%m-%d") for value in table["date"].to_pylist()]
        current_securities = table.column_names[1:]
        if not current_dates or current_dates != sorted(set(current_dates)):
            raise ValueError(f"{metric} 的有效区间需要唯一、递增的交易日期")
        if dates is None:
            dates, securities = current_dates, current_securities
            digest.update("\n".join(dates + securities).encode())
        elif dates != current_dates or securities != current_securities:
            raise ValueError(f"{metric} 与参考矩阵的日期或证券顺序存在差异")
        if any(table.schema.field(s).type != pa.float64() for s in current_securities):
            raise ValueError(f"{metric} 的证券字段应采用 float64")
        values = np.ascontiguousarray(table.select(current_securities).to_pandas().to_numpy(), dtype=np.float64)
        values.setflags(write=False)
        arrays[metric] = values
        digest.update(metric.encode())
        digest.update(values.tobytes())
        nulls[metric] = sum(column.null_count for column in table.columns[1:])
        files.append({"metric": metric, "file": str(path), "size_bytes": path.stat().st_size,
                      "sha256": sha256(path), "loaded_rows": len(dates), "loaded_columns": len(securities)})
        progress(stage="data", done=number + 1, total=len(METRICS), message=f"读取 {metric}")
    if config.start not in dates or config.end not in dates:
        raise ValueError("起止日期应对应输入矩阵中的交易日")
    if dates[0] >= config.start:
        raise ValueError("请为首日委托提供至少一个起始日前的交易日")
    return MarketData(dates, securities, arrays, digest.hexdigest(), files, nulls)


class History:
    """所有策略读取入口都以当前收盘为边界。"""

    def __init__(self, data: MarketData, features, status, row: int, eligible: np.ndarray, reasons: np.ndarray, overrides=None, announcements=()):
        self._data, self._features = data, features
        self._row = row
        self.date = data.dates[row]
        self.securities = tuple(data.securities)
        self.eligible = eligible.copy()
        self.reasons = reasons.copy()
        self.status = status.values[row].copy()
        self.announcements = tuple(announcements)
        for security, record in (overrides or {}).items():
            if security in data.index and "is_st" in record:
                self.status[data.index[security]] = int(record["is_st"])

    def prices(self, metric="close", periods=None, adjusted=False):
        if metric not in METRICS:
            raise ValueError("请选择已登记的行情字段")
        start = 0 if periods is None else max(0, self._row + 1 - periods)
        values = self._data.arrays[metric][start:self._row + 1].copy()
        if adjusted and metric in {"open", "high", "low", "close"}:
            values *= self._data.arrays["cumadj"][start:self._row + 1]
        return values

    def factor(self, name):
        return self._features.values[name][self._row].copy()

    @property
    def available_dates(self):
        return tuple(self._data.dates[:self._row + 1])


class FactorProvider:
    """具有生效时间与可获知时间的扩展因子。"""

    def __init__(self, records: pd.DataFrame):
        required = {"security", "factor", "value", "effective_at", "known_at"}
        if not required <= set(records):
            raise ValueError("扩展因子需要 security、factor、value、effective_at、known_at")
        self._records = records.copy()
        from .events import timestamp
        for key in ("effective_at", "known_at"):
            self._records[key] = pd.to_datetime(self._records[key].map(timestamp), utc=True)

    def as_of(self, when: str) -> pd.DataFrame:
        now = pd.Timestamp(when)
        now = now.tz_localize("Asia/Shanghai") if now.tzinfo is None else now
        frame = self._records[(self._records.known_at <= now) & (self._records.effective_at <= now)]
        return frame.sort_values(["known_at", "effective_at"]).drop_duplicates(["security", "factor"], keep="last").copy()
