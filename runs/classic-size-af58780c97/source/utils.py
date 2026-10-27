from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
import hashlib
import json
import os

CENT = Decimal("0.01")


def dec(value) -> Decimal:
    return value if isinstance(value, Decimal) else Decimal(str(value))


def money(value) -> Decimal:
    return dec(value).quantize(CENT, rounding=ROUND_HALF_UP)


def json_default(value):
    if isinstance(value, (Decimal, Path, date, datetime)):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(type(value).__name__)


def dumps(value, **kwargs) -> str:
    if kwargs.pop("pretty", False):
        kwargs.setdefault("indent", 2)
    return json.dumps(value, ensure_ascii=False, default=json_default, allow_nan=False, **kwargs)


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(value) -> str:
    return hashlib.sha256(dumps(value, sort_keys=True).encode()).hexdigest()


def contained(root: Path, path: Path) -> Path:
    path = Path(path)
    path = (root / path if not path.is_absolute() else path).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("文件路径应位于项目目录内")
    return path


class Cancelled(Exception):
    """用户在可重放边界结束本次任务。"""


def check_cancel(path: Path | None):
    if path is not None and path.exists():
        raise Cancelled("已在安全边界取消，已生成的记录保留供检查")

