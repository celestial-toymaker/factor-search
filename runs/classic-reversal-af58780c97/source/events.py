from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import json

from .utils import fingerprint

SHANGHAI = ZoneInfo("Asia/Shanghai")
PHASES = {"initialize": "08:00:00", "pre_open": "09:00:00", "open": "09:30:00",
          "close": "15:00:00", "rights": "15:01:00", "decision": "15:02:00", "settlement": "15:05:00", "end": "15:10:00"}
ACTION_ORDER = {"security_status": 0, "rule_change": 1, "code_change": 2, "split": 3, "stock_dividend": 4,
                "dividend_ex": 5, "dividend_pay": 6, "cash_deposit": 7, "cash_withdrawal": 8,
                "rights_subscribe": 9, "conversion": 10, "spinoff": 11, "delist_settlement": 12,
                "tax": 13, "fee": 14, "order_cancel": 15, "order_replace": 16, "dividend_record": 20}
CORPORATE_TYPES = {"dividend_record", "dividend_ex", "dividend_pay", "split", "stock_dividend", "rights_offer",
                   "rights_subscribe", "conversion", "merger", "spinoff", "delist_settlement", "fraction_settlement"}


def timestamp(value) -> datetime:
    if isinstance(value, datetime):
        result = value
    else:
        result = datetime.fromisoformat(str(value))
    return result.replace(tzinfo=SHANGHAI) if result.tzinfo is None else result.astimezone(SHANGHAI)


def at(day, phase):
    return timestamp(day + "T" + PHASES.get(phase, phase))


@dataclass(frozen=True)
class Event:
    event_id: str
    type: str
    effective_at: datetime
    known_at: datetime
    security: str | None = None
    payload: dict = field(default_factory=dict)
    source: str = "user"

    def __post_init__(self):
        object.__setattr__(self, "effective_at", timestamp(self.effective_at))
        object.__setattr__(self, "known_at", timestamp(self.known_at))
        if not self.event_id or not self.type or not self.source:
            raise ValueError("事件需要唯一编号、类型和来源")

    @property
    def ready_at(self):
        return max(self.effective_at, self.known_at)

    def as_dict(self):
        return {"event_id": self.event_id, "type": self.type, "security": self.security,
                "effective_at": self.effective_at.isoformat(), "known_at": self.known_at.isoformat(),
                "payload": self.payload, "source": self.source}


class EventSource:
    def __init__(self, events=()):
        distinct = {}
        for event in events:
            event = event if isinstance(event, Event) else Event(**event)
            if event.event_id in distinct and fingerprint(distinct[event.event_id].as_dict()) != fingerprint(event.as_dict()):
                raise ValueError(f"事件编号对应不同内容：{event.event_id}")
            distinct[event.event_id] = event
        self.events = sorted(distinct.values(), key=lambda e: (e.ready_at, ACTION_ORDER.get(e.type, 18), e.security or "", e.event_id))
        self._cursor = 0

    @classmethod
    def from_jsonl(cls, path: Path | None):
        if path is None:
            return cls()
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        expanded = []
        for record in records:
            if record["type"] == "dividend":
                payload = record["payload"]
                for kind, key, phase in [("dividend_record", "record_date", "rights"),
                                         ("dividend_ex", "ex_date", "pre_open"),
                                         ("dividend_pay", "pay_date", "pre_open")]:
                    expanded.append({**record, "event_id": record["event_id"] + ":" + kind,
                                     "type": kind, "effective_at": at(payload[key], phase),
                                     "payload": {**payload, "action_id": record["event_id"]}})
            else:
                expanded.append(record)
        return cls(expanded)

    def due(self, now):
        result = []
        while self._cursor < len(self.events) and self.events[self._cursor].ready_at <= now:
            result.append(self.events[self._cursor])
            self._cursor += 1
        return result

    def known(self, now):
        return tuple(event for event in self.events if event.known_at <= now)
