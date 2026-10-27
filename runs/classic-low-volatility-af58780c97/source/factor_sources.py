"""免费公开基本面数据的批量获取、原始响应缓存与来源清单。"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import gzip
import json
import time
import urllib.parse
import urllib.request

import pandas as pd

from .utils import atomic_json, fingerprint, sha256

ENDPOINT = "https://datacenter-web.eastmoney.com/api/data/v1/get"


@dataclass(frozen=True)
class SourceQuery:
    kind: str
    report: str
    filter: str
    columns: str = "ALL"

    @property
    def key(self):
        return fingerprint(self.__dict__)[:24]


class PublicDataCache:
    def __init__(self, directory: Path, timeout=25, attempts=4):
        self.directory = Path(directory)
        self.timeout, self.attempts = timeout, attempts

    def fetch(self, query: SourceQuery):
        directory = self.directory / query.kind / query.key
        ready = directory / "manifest.json"
        if ready.exists():
            manifest = json.loads(ready.read_text(encoding="utf-8"))
            if manifest.get("query") == query.__dict__ and all(
                (directory / page["file"]).exists()
                and sha256(directory / page["file"]) == page["sha256"]
                for page in manifest["pages"]
            ):
                return manifest
        directory.mkdir(parents=True, exist_ok=True)
        pages, count, expected = [], 0, None
        page_number = 1
        while True:
            params = {"reportName": query.report, "columns": query.columns,
                      "filter": query.filter, "pageSize": "10000", "pageNumber": str(page_number),
                      "sortColumns": "SECUCODE", "sortTypes": "1", "source": "WEB", "client": "WEB"}
            url = ENDPOINT + "?" + urllib.parse.urlencode(params)
            payload = self._request(url)
            result = payload.get("result")
            if result is None:
                raise RuntimeError(f"{query.kind} 返回结构需要核验：{payload.get('message')}")
            records = result.get("data") or []
            expected = int(result.get("count", len(records)))
            count += len(records)
            path = directory / f"page-{page_number:04d}.json.gz"
            encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
            temporary = path.with_suffix(".tmp")
            temporary.write_bytes(gzip.compress(encoded, mtime=0))
            temporary.replace(path)
            pages.append({"file": path.name, "sha256": sha256(path), "rows": len(records), "url": url})
            if page_number >= int(result.get("pages", 1)):
                break
            page_number += 1
        if count != expected:
            raise RuntimeError(f"{query.kind} 分页数量需要核验：{count} / {expected}")
        manifest = {"query": query.__dict__, "key": query.key, "directory": str(directory),
                    "retrieved_at": datetime.now(timezone.utc).isoformat(), "rows": count, "pages": pages}
        atomic_json(ready, manifest)
        return manifest

    def _request(self, url):
        last = None
        for attempt in range(self.attempts):
            try:
                request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"})
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    payload = json.load(response)
                if payload.get("success") is True:
                    return payload
                raise RuntimeError(payload.get("message", "数据源返回状态需要核验"))
            except Exception as exc:
                last = exc
                if attempt + 1 < self.attempts:
                    time.sleep(min(2 ** attempt, 6))
        raise RuntimeError(f"免费数据请求失败：{last}") from last

    @staticmethod
    def read(manifest):
        rows = []
        directory = Path(manifest["directory"])
        for page in manifest["pages"]:
            path = directory / page["file"]
            if sha256(path) != page["sha256"]:
                raise ValueError("原始响应哈希需要核验：" + str(path))
            payload = json.loads(gzip.decompress(path.read_bytes()))
            rows.extend(payload["result"]["data"])
        return rows


def source_queries(dates, financial_start, end):
    queries = [SourceQuery("shares", "RPT_VALUEANALYSIS_DET", f"(TRADE_DATE='{day}')",
                           "SECUCODE,TRADE_DATE,TOTAL_SHARES,TOTAL_MARKET_CAP,CLOSE_PRICE") for day in dates]
    for quarter in pd.date_range(financial_start, end, freq="QE-DEC"):
        day = quarter.strftime("%Y-%m-%d")
        if day >= end:
            continue
        reports = [("balance", f"RPT_F10_FINANCE_{prefix}BALANCE") for prefix in ("G", "B", "I", "S")]
        reports.append(("income", "RPT_F10_FINANCE_GINCOME"))
        for kind, report in reports:
            fields = "SECUCODE,REPORT_DATE,NOTICE_DATE,UPDATE_DATE,CURRENCY,"
            fields += "TOTAL_ASSETS,TOTAL_PARENT_EQUITY" if kind == "balance" else "TOTAL_OPERATE_INCOME,PARENT_NETPROFIT"
            queries.append(SourceQuery(kind, report, f"(REPORT_DATE='{day}')", fields))
    queries.append(SourceQuery("dividend", "RPT_SHAREBONUS_DET",
                               f"(EX_DIVIDEND_DATE>='{pd.Timestamp(dates[0])-pd.Timedelta(days=366):%Y-%m-%d}')(EX_DIVIDEND_DATE<='{end}')"))
    return queries


def fetch_sources(config, dates, progress=lambda **kw: None, *, kinds=None):
    root = config.root / "data" / "classic_factors"
    cache = PublicDataCache(root / "raw")
    queries = source_queries(dates, config.parameters.get("financial_start", "2022-01-01"), config.end)
    if kinds:
        queries = [query for query in queries if query.kind in kinds]
    manifests, errors = [], []
    workers = int(config.parameters.get("source_workers", 4))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(cache.fetch, query): query for query in queries}
        for number, future in enumerate(as_completed(futures), 1):
            query = futures[future]
            try:
                manifest = future.result()
                manifests.append(manifest)
            except Exception as exc:
                errors.append({"query": query.__dict__, "error": str(exc)})
            if number % 5 == 0 or number == len(queries):
                progress(stage="sources", done=number, total=len(queries),
                         message=f"免费数据缓存 {number}/{len(queries)}，待补请求 {len(errors)}")
                atomic_json(root / "source_progress.json", {"done": number, "total": len(queries), "errors": errors})
    manifests.sort(key=lambda item: (item["query"]["kind"], item["key"]))
    result = {"manifests": manifests, "errors": errors, "requested": len(queries),
              "fingerprint": fingerprint([{k: m[k] for k in ("query", "key", "pages")} for m in manifests])}
    atomic_json(root / "sources.json", result)
    return result
