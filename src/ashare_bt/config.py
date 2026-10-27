from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from datetime import date
from decimal import Decimal
from pathlib import Path
import json
import tomllib

ROOT = Path(__file__).resolve().parents[2]
EXIT_MODES = {
    "latest_feasible": "按规则最晚启动退出",
    "force_cash": "末日强制结清",
    "strict_end": "末日按规则退出",
}


@dataclass
class Config:
    name: str = "非 ST 全市场动量 · 10 亿"
    root: Path = field(default_factory=lambda: ROOT)
    data_dir: str = "feathers"
    status_dir: str = "data/status"
    output_dir: str = "runs"
    start: str = "2025-01-02"
    end: str = "2026-09-30"
    warmup_start: str = "2024-01-01"
    initial_cash: str = "1000000000.00"
    device: str = "auto"
    block_days: int = 128
    accounting_mode: str = "total_return"
    exit_mode: str = "latest_feasible"
    strategy: str = "momentum"
    rebalance_days: int = 5
    max_positions: int = 20
    target_exposure: str = "0.95"
    min_history: int = 60
    min_adv20: str = "20000000"
    participation: str = "0.01"
    commission_rate: str = "0.0001"
    min_commission: str = "5.00"
    stamp_tax_rate: str = "0.0005"
    slippage_rate: str = "0.0005"
    auto_sync_status: bool = True
    strict_status: bool = True
    events_file: str | None = None
    status_file: str | None = None
    parameters: dict = field(default_factory=dict)

    def __post_init__(self):
        self.root = Path(self.root).resolve()
        for key in ("warmup_start", "start", "end"):
            date.fromisoformat(getattr(self, key))
        if not self.warmup_start <= self.start <= self.end:
            raise ValueError("日期应满足：预热开始 ≤ 交易开始 ≤ 交易结束")
        if self.device not in {"auto", "mps", "cuda", "reference"}:
            raise ValueError("计算设备请选择 auto、mps 或 cuda；reference 用于数值验证")
        if self.exit_mode not in EXIT_MODES:
            raise ValueError("请选择有效的退出方式")
        if self.accounting_mode not in {"total_return", "explicit"}:
            raise ValueError("公司行动口径请选择 total_return 或 explicit")
        if not 1 <= self.block_days <= 512:
            raise ValueError("分块天数应在 1 至 512 之间")
        if not 1 <= self.max_positions <= 5451 or self.rebalance_days < 1:
            raise ValueError("持仓数量与调仓间隔应为正整数")
        for key in ("initial_cash", "min_history", "min_adv20"):
            if Decimal(str(getattr(self, key))) <= 0:
                raise ValueError(f"{key} 应大于零")
        for key in ("target_exposure", "participation"):
            if not 0 < Decimal(str(getattr(self, key))) <= 1:
                raise ValueError(f"{key} 应在 0 至 1 之间")
        for key in ("commission_rate", "min_commission", "stamp_tax_rate", "slippage_rate"):
            if Decimal(str(getattr(self, key))) < 0:
                raise ValueError(f"{key} 应为零或正数")
        for key in ("window", "fast", "slow", "exit_window", "rsi_window"):
            if key in self.parameters and not 2 <= int(self.parameters[key]) <= 252:
                raise ValueError(f"{key} 应在 2 至 252 个交易日之间")
        if int(self.parameters.get("fast", 20)) >= int(self.parameters.get("slow", 60)):
            raise ValueError("短均线周期应小于长均线周期")
        if self.parameters.get("allocation", "equal") not in {"equal", "inverse_volatility"}:
            raise ValueError("仓位分配请选择等权或逆波动率")

    def path(self, key: str) -> Path:
        path = Path(getattr(self, key))
        return path if path.is_absolute() else self.root / path

    def public(self) -> dict:
        return {k: v for k, v in asdict(self).items() if k != "root"}

    @classmethod
    def from_dict(cls, values: dict, root: Path | None = None):
        values = dict(values)
        values.pop("benchmark", None)  # 兼容历史运行保存的配置。
        allowed = {f.name for f in fields(cls)}
        unknown = set(values) - allowed
        if unknown:
            raise ValueError("请检查配置字段：" + ", ".join(sorted(unknown)))
        return cls(**{**values, "root": root or values.get("root", ROOT)})

    @classmethod
    def load(cls, path: str | Path | None = None):
        if path is None:
            return cls()
        path = Path(path).resolve()
        content = path.read_text(encoding="utf-8")
        values = json.loads(content) if path.suffix == ".json" else tomllib.loads(content)
        return cls.from_dict(values, path.parent.parent if path.parent.name == "configs" else ROOT)

