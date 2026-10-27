from __future__ import annotations

from datetime import datetime
from pathlib import Path
import hashlib
import json
import time
import copy

import numpy as np
import pandas as pd

from .utils import Cancelled, atomic_json, check_cancel, sha256


def _status_login():
    import baostock as bs
    import socket
    import os
    socket.setdefaulttimeout(90)
    time.sleep((os.getpid() % 4) * 1.5)
    for attempt in range(5):
        login = bs.login()
        if login.error_code == "0":
            return
        time.sleep(2 + attempt)
    raise RuntimeError("历史状态数据源连接失败：" + login.error_msg)


def _fetch_status_day(day, directory):
    import baostock as bs
    path = Path(directory) / "daily" / f"{day}.parquet"
    if path.exists():
        return {"date": day}
    for attempt in range(3):
        try:
            result = bs.query_daily_history_k_AStock(day)
            rows = []
            while result.error_code == "0" and result.next():
                rows.append(result.get_row_data())
            if result.error_code != "0" or not rows:
                raise RuntimeError(result.error_msg or "返回空快照")
            frame = pd.DataFrame(rows, columns=result.fields)
            if not {"date", "code", "isST", "tradestatus"} <= set(frame):
                raise RuntimeError("逐日快照需要 isST 与 tradestatus 字段")
            if set(frame.date) != {day}:
                raise RuntimeError("源快照日期需要与请求日期一致")
            frame["security"] = frame.code.str.slice(3) + "." + frame.code.str.slice(0, 2).str.upper()
            frame["is_st"] = frame.isST
            frame["trade_status"] = frame.tradestatus
            frame["source"] = "BaoStock 0.9.4 / query_daily_history_k_AStock"
            frame["known_at"] = frame.date + "T09:00:00+08:00"
            frame["availability_basis"] = "交易日风险状态的开盘前生效推定"
            temporary = path.with_suffix(".tmp.parquet")
            frame.to_parquet(temporary, index=False)
            temporary.replace(path)
            return {"date": day}
        except Exception as exc:
            if attempt == 2:
                return {"date": day, "error": str(exc)}
            time.sleep(0.5 * (attempt + 1))


def baostock_code(security: str) -> str:
    code, exchange = security.split(".")
    return exchange.lower() + "." + code


def sync_status(securities, first: str, last: str, directory: Path,
                progress=lambda **kw: None, cancel_file: Path | None = None,
                limit: int | None = None, dates: list[str] | None = None):
    """保留逐证券原始状态快照，支持逐个文件续传。"""
    import baostock as bs
    directory.mkdir(parents=True, exist_ok=True)
    if dates is not None and limit is None:
        return sync_daily_status(dates, directory, progress, cancel_file)
    selected = list(securities)[:limit] if limit else list(securities)
    pending = []
    for security in selected:
        path = directory / f"{security}_{first}_{last}.parquet"
        if not path.exists():
            pending.append((security, path))
    progress(stage="status", done=len(selected) - len(pending), total=len(selected), message="准备逐日 ST 状态")
    if not pending:
        return
    login = bs.login()
    if login.error_code != "0":
        raise RuntimeError("历史状态数据源连接失败：" + login.error_msg)
    errors = []
    try:
        for number, (security, path) in enumerate(pending):
            check_cancel(cancel_file)
            last_error = ""
            for attempt in range(3):
                try:
                    result = bs.query_history_k_data_plus(
                        baostock_code(security), "date,code,isST,tradestatus",
                        start_date=first, end_date=last, frequency="d", adjustflag="3")
                    records = []
                    while result.error_code == "0" and result.next():
                        records.append(result.get_row_data())
                    if result.error_code != "0":
                        raise RuntimeError(result.error_msg)
                    frame = pd.DataFrame(records, columns=["date", "source_code", "is_st", "trade_status"])
                    frame["security"] = security
                    frame["source"] = "BaoStock 0.9.4 / query_history_k_data_plus / isST"
                    frame["known_at"] = frame["date"] + "T09:00:00+08:00"
                    frame["availability_basis"] = "交易日风险状态的开盘前生效推定"
                    temporary = path.with_suffix(".tmp.parquet")
                    frame.to_parquet(temporary, index=False)
                    temporary.replace(path)
                    last_error = ""
                    break
                except Exception as exc:
                    last_error = str(exc)
                    if attempt < 2:
                        time.sleep(0.3 * (attempt + 1))
            if last_error:
                errors.append({"security": security, "error": last_error})
            done = len(selected) - len(pending) + number + 1
            if done % 25 == 0 or number == len(pending) - 1:
                progress(stage="status", done=done, total=len(selected), message=f"历史状态 {done:,} / {len(selected):,}")
                atomic_json(directory / "sync_progress.json", {"done": done, "total": len(selected), "errors": errors,
                            "first": first, "last": last})
    finally:
        bs.logout()
    if errors:
        atomic_json(directory / "sync_errors.json", errors)


class StatusProvider:
    """逐日状态与证据；未知为 -1、正常为 0、ST/*ST 为 1。"""

    def __init__(self, dates, securities, directory: Path, local_file: Path | None = None):
        self.dates = list(dates)
        self.securities = list(securities)
        self.values = np.full((len(dates), len(securities)), -1, dtype=np.int8)
        self.open_values = self.values.copy()
        self.trade_status = np.full_like(self.values, -1)
        date_index = {day: i for i, day in enumerate(dates)}
        security_index = {security: j for j, security in enumerate(securities)}
        files = sorted(directory.glob("daily/*.parquet")) + sorted(directory.glob("*.parquet")) if directory.exists() else []
        files = [path for path in files if ".tmp." not in path.name]
        self.sources = []
        self.overrides = {}
        for path in files:
            if path.suffix == ".parquet":
                frame = pd.read_parquet(path)
            elif path.suffix == ".jsonl":
                frame = pd.read_json(path, lines=True, dtype=False)
            else:
                frame = pd.read_csv(path, dtype=str)
            if not {"date", "security", "is_st"} <= set(frame):
                raise ValueError(f"状态文件需要 date、security、is_st：{path.name}")
            source_name = str(path.relative_to(directory)) if path.is_relative_to(directory) else path.name
            self.sources.append({"file": source_name, "sha256": sha256(path), "rows": len(frame)})
            for row in frame.to_dict("records"):
                day = str(row["date"])[:10]
                i, j = date_index.get(day), security_index.get(row["security"])
                if i is None or j is None:
                    continue
                flag = str(row["is_st"]).strip().lower()
                if flag not in {"0", "0.0", "false", "1", "1.0", "true"}:
                    continue
                value = int(flag in {"1", "1.0", "true"})
                known = str(row.get("known_at", day + "T09:00:00+08:00"))
                if known <= day + "T15:00:00+08:00":
                    self.values[i, j] = value
                if known <= day + "T09:30:00+08:00":
                    self.open_values[i, j] = value
                trade = str(row.get("trade_status", "-1"))
                if trade in {"0", "1"}:
                    self.trade_status[i, j] = int(trade)
        if local_file:
            from .events import timestamp, at
            frame = (pd.read_parquet(local_file) if local_file.suffix == ".parquet" else
                     pd.read_json(local_file, lines=True, dtype=False) if local_file.suffix == ".jsonl" else
                     pd.read_csv(local_file, dtype=str))
            if not {"date", "security", "is_st"} <= set(frame):
                raise ValueError("本地状态需要 date、security、is_st 字段")
            self.sources.append({"file": local_file.name, "sha256": sha256(local_file), "rows": len(frame), "priority": "override"})
            records = frame.to_dict("records")
            records.sort(key=lambda r: max(timestamp(r.get("effective_at", str(r["date"])[:10] + "T09:00:00")),
                                           timestamp(r.get("known_at", str(r["date"])[:10] + "T09:00:00"))))
            for record in records:
                j = security_index.get(record["security"])
                if j is None:
                    continue
                effective = timestamp(record.get("effective_at", str(record["date"])[:10] + "T09:00:00"))
                ready = max(effective, timestamp(record.get("known_at", effective)))
                value = self.parse_flag(record["is_st"])
                for i, day in enumerate(dates):
                    if ready <= at(day, "close"):
                        self.values[i, j] = value
                    if ready <= at(day, "open"):
                        self.open_values[i, j] = value
        self.fingerprint = hashlib.sha256(json.dumps(self.sources, sort_keys=True).encode()).hexdigest()

    @staticmethod
    def parse_flag(flag):
        value = str(flag).strip().lower()
        if value in {"1","1.0","true"}:
            return 1
        if value in {"0","0.0","false"}:
            return 0
        if value in {"-1","-1.0","unknown","none","nan"}:
            return -1
        raise ValueError("is_st 请选择 0（普通）、1（ST）或 -1（待核实）")

    def with_events(self, events):
        from .events import at
        state_events = [event for event in events if event.type=="security_status" and "is_st" in event.payload]
        if not state_events:
            return self
        result = copy.copy(self)
        result.values = self.values.copy()
        result.open_values = self.open_values.copy()
        indices = {security:j for j,security in enumerate(self.securities)}
        for event in state_events:
            column = indices.get(event.security)
            if column is None:
                continue
            value = self.parse_flag(event.payload["is_st"])
            for row,day in enumerate(self.dates):
                if event.ready_at <= at(day,"close"):
                    result.values[row,column] = value
                if event.ready_at <= at(day,"open"):
                    result.open_values[row,column] = value
        result.sources = [*self.sources,{"state_events":[event.as_dict() for event in state_events]}]
        result.fingerprint = hashlib.sha256(json.dumps(result.sources,sort_keys=True,default=str).encode()).hexdigest()
        return result

    @classmethod
    def synthetic(cls, dates, securities, values=None):
        obj = cls(dates, securities, Path("/private/tmp/ashare-empty-status-source"))
        obj.values[:] = 0 if values is None else values
        obj.open_values[:] = obj.values
        obj.fingerprint = "synthetic-fixture"
        return obj

    def coverage(self, required: np.ndarray):
        unknown = required & (self.values < 0)
        locations = np.argwhere(unknown)
        return {
            "required_stock_days": int(required.sum()),
            "covered_stock_days": int((required & (self.values >= 0)).sum()),
            "unknown_stock_days": int(unknown.sum()),
            "unknown_examples": [{"date": self.dates[i], "security": self.securities[j]} for i, j in locations[:30]],
            "st_stock_days": int((required & (self.values == 1)).sum()),
            "source_fingerprint": self.fingerprint,
            "availability_basis": "逐日风险状态按当日开盘前生效推定；有明确时间的覆盖事件按其可获知时间执行",
        }


def sync_daily_status(dates, directory, progress=lambda **kw: None, cancel_file=None):
    """逐日保存源接口的完整行，并补充统一状态字段。"""
    from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
    import multiprocessing
    daily = directory / "daily"
    daily.mkdir(parents=True, exist_ok=True)
    pending = [day for day in dates if not (daily / f"{day}.parquet").exists()]
    if not pending:
        progress(stage="status", done=len(dates), total=len(dates), message="历史状态快照已就绪")
        return
    errors = []
    with ProcessPoolExecutor(max_workers=min(3, len(pending)), mp_context=multiprocessing.get_context("spawn"),
                             initializer=_status_login) as executor:
        futures = {executor.submit(_fetch_status_day, day, str(directory)): day for day in pending}
        remaining = set(futures)
        completed = len(dates) - len(pending)
        while remaining:
            try:
                check_cancel(cancel_file)
            except Cancelled:
                for process in executor._processes.values():
                    process.terminate()
                for future in remaining:
                    future.cancel()
                raise
            ready, remaining = wait(remaining, timeout=.5, return_when=FIRST_COMPLETED)
            for future in ready:
                record = future.result()
                day = record["date"]
                if "error" in record:
                    errors.append(record)
                completed += 1
                progress(stage="status", done=completed, total=len(dates), message=f"历史状态快照 {completed} / {len(dates)}")
                atomic_json(directory / "sync_progress.json", {"done": completed, "total": len(dates), "last_date": day, "errors": errors})
    if errors:
        atomic_json(directory / "sync_errors.json", errors)


def ensure_status(data, config, progress=lambda **kw: None, cancel_file=None):
    indices = data.trading_indices(config)
    first = data.dates[indices[0] - 1]
    dates = data.dates[indices[0] - 1:indices[-1] + 1]
    directory = config.path("status_dir")
    if config.auto_sync_status:
        sync_status(data.securities, first, config.end, directory, progress, cancel_file, dates=dates)
    local = config.path("status_file") if config.status_file else None
    provider = StatusProvider(data.dates, data.securities, directory, local)
    raw_required = (np.isfinite(data.arrays["close"]) & np.isfinite(data.arrays["cumadj"]) &
                    (data.arrays["amt"] > 0) & data.stock_mask[None, :])
    raw_required[:indices[0] - 1] = False
    missing = raw_required & (provider.values < 0)
    if config.auto_sync_status and missing.any():
        securities = [data.securities[j] for j in np.flatnonzero(missing.any(axis=0))]
        progress(stage="status", done=0, total=len(securities), message="补核证券代码与逐日快照差异")
        sync_status(securities, first, config.end, directory, progress, cancel_file)
        provider = StatusProvider(data.dates, data.securities, directory, local)
    return provider
