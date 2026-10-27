from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import platform
import time
from contextlib import contextmanager

import numpy as np
import torch

from .config import Config
from .data import METRICS, MarketData
from .utils import check_cancel, fingerprint


def select_device(requested="auto") -> torch.device:
    if requested == "reference":
        return torch.device("cpu")
    if requested == "auto":
        if platform.system() == "Darwin" and torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        raise RuntimeError("请在具有 GPU 访问权限的环境运行：Apple Silicon 使用 MPS，NVIDIA 使用 CUDA")
    if requested == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    raise RuntimeError(f"请检查 {requested.upper()} 的硬件访问、驱动及 PyTorch 安装")


def synchronize(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def device_name(device):
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    if device.type == "mps":
        import ctypes
        try:
            library = ctypes.CDLL(None)
            size = ctypes.c_size_t()
            if library.sysctlbyname(b"machdep.cpu.brand_string", None, ctypes.byref(size), None, 0) == 0:
                buffer = ctypes.create_string_buffer(size.value)
                library.sysctlbyname(b"machdep.cpu.brand_string", buffer, ctypes.byref(size), None, 0)
                return buffer.value.decode()
        except (OSError, AttributeError):
            pass
    return platform.machine()


@contextmanager
def gpu_lock(config, progress, cancel_file):
    import fcntl
    path = config.root / "data" / "gpu.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        waiting = False
        while True:
            check_cancel(cancel_file)
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if not waiting:
                    progress(stage="gpu_queue", done=0, total=1, message="等待本机已有 GPU 计算完成")
                    waiting = True
                time.sleep(.2)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def rolling(values: torch.Tensor, window: int, operation="mean"):
    result = torch.full_like(values, float("nan"))
    if values.shape[0] < window:
        return result
    view = values.unfold(0, window, 1)
    if operation == "mean":
        reduced = view.mean(dim=-1)
    elif operation == "std":
        reduced = view.std(dim=-1, correction=0)
    elif operation == "max":
        reduced = view.amax(dim=-1)
    elif operation == "min":
        reduced = view.amin(dim=-1)
    else:
        raise ValueError(operation)
    result[window - 1:] = reduced
    return result


@dataclass
class FeatureBundle:
    values: dict[str, np.ndarray]
    flags: dict[str, np.ndarray]
    metadata: dict
    profile: dict


def compute_features(data: MarketData, config: Config, progress=lambda **kw: None,
                     cancel_file: Path | None = None, use_cache=True) -> FeatureBundle:
    if config.device == "reference":
        return _compute_features(data, config, progress, cancel_file, use_cache)
    select_device(config.device)
    with gpu_lock(config, progress, cancel_file):
        return _compute_features(data, config, progress, cancel_file, use_cache)


def _compute_features(data: MarketData, config: Config, progress=lambda **kw: None,
                     cancel_file: Path | None = None, use_cache=True) -> FeatureBundle:
    device = select_device(config.device)
    params = config.parameters
    windows = sorted({5, 10, 14, 20, 60, config.min_history,
                      *(int(params[key]) for key in ("window", "fast", "slow", "exit_window", "rsi_window") if key in params)})
    histories = sorted({config.min_history, max(windows)})
    rsi_windows = sorted({14, int(params.get("rsi_window", params.get("window", 14) if config.strategy == "rsi" else 14))})
    key = fingerprint({"schema": 6, "data": data.fingerprint, "windows": windows, "histories": histories, "rsi_windows": rsi_windows, "hardware":device_name(device),
                       "device": str(device), "torch": torch.__version__, "block": config.block_days})
    cache = config.root / "data" / "features" / f"{key}.npz"
    if use_cache and cache.exists():
        with np.load(cache, allow_pickle=False) as source:
            meta = json.loads(str(source["__metadata__"]))
            profile = json.loads(str(source["__profile__"]))
            values = {name[2:]: source[name] for name in source.files if name.startswith("v_")}
            flags = {name[2:]: source[name] for name in source.files if name.startswith("f_")}
        progress(stage="factors", done=data.shape[0], total=data.shape[0], message="复用已核验的 GPU 因子缓存")
        profile = {**profile,"dates":{"first_loaded":data.dates[0],"last_loaded":data.dates[-1],
                                      "start":config.start,"end":config.end,"trading_days":len(data.trading_indices(config))},
                   "source_files":data.source_files,"fingerprint":data.fingerprint}
        return FeatureBundle(values, flags, {**meta, "cache_hit": True}, profile)
    rows, columns = data.shape
    names = {"ret_1", "adv_20", "history_valid"}
    for window in windows:
        names.update(f"{prefix}_{window}" for prefix in ("mom", "ma", "vol", "std", "high_prev", "low_prev"))
    names.update(f"rsi_{window}" for window in rsi_windows)
    values = {name: np.full(data.shape, np.nan, np.float32) for name in names}
    flags = {name: np.zeros(data.shape, dtype=bool) for name in
             ("valid_ohlc", "zero_amount", "invalid_ohlc", "factor_change", "small_factor_change")}
    counts = {metric: {"missing": 0, "infinite": 0, "zero": 0, "negative": 0} for metric in METRICS}
    previous_factor = torch.full((columns,), float("nan"), device=device)
    previous_factor_words = torch.zeros((columns, 2), dtype=torch.int32, device=device)
    rsi_state = {window: {"gain": torch.zeros(columns, device=device), "loss": torch.zeros(columns, device=device),
                         "count": torch.zeros(columns, dtype=torch.int32, device=device)} for window in rsi_windows}
    started = time.monotonic()
    halo = max(windows) + 2
    for start in range(0, rows, config.block_days):
        check_cancel(cancel_file)
        stop = min(start + config.block_days, rows)
        lower = max(0, start - halo)
        offset = start - lower
        tensors = {metric: torch.from_numpy(np.ascontiguousarray(data.arrays[metric][lower:stop], dtype=np.float32)).to(device)
                   for metric in METRICS}
        factor_words = torch.from_numpy(np.array(data.arrays["cumadj"][lower:stop], copy=True, order="C").view(np.int32).reshape(stop-lower, columns, 2)).to(device)
        def save(name, tensor):
            values[name][start:stop] = tensor[offset:].detach().cpu().numpy()
        adjusted = tensors["close"] * tensors["cumadj"]
        high = tensors["high"] * tensors["cumadj"]
        low = tensors["low"] * tensors["cumadj"]
        valid = torch.isfinite(adjusted) & (adjusted > 0)
        returns = torch.full_like(adjusted, float("nan"))
        returns[1:] = adjusted[1:] / adjusted[:-1] - 1
        save("ret_1", returns)
        save("adv_20", rolling(tensors["amt"], 20))
        save("history_valid", rolling(valid.float(), config.min_history, "min"))
        for window in windows:
            momentum = torch.full_like(adjusted, float("nan"))
            if len(adjusted) > window:
                momentum[window:] = adjusted[window:] / adjusted[:-window] - 1
            save(f"mom_{window}", momentum)
            save(f"ma_{window}", rolling(adjusted, window))
            save(f"std_{window}", rolling(adjusted, window, "std"))
            save(f"vol_{window}", rolling(returns, window, "std"))
            prev_high = torch.full_like(high, float("nan"))
            prev_low = torch.full_like(low, float("nan"))
            prev_high[1:] = rolling(high, window, "max")[:-1]
            prev_low[1:] = rolling(low, window, "min")[:-1]
            save(f"high_prev_{window}", prev_high)
            save(f"low_prev_{window}", prev_low)
        for metric, tensor in tensors.items():
            block = tensor[offset:]
            tests = (torch.isnan(block), torch.isinf(block), block == 0, block < 0)
            for label, mask in zip(counts[metric], tests):
                counts[metric][label] += int(mask.sum(dtype=torch.int32).item())
        good_prices = torch.ones_like(valid)
        for metric in ("open", "high", "low", "close"):
            good_prices &= torch.isfinite(tensors[metric]) & (tensors[metric] > 0)
        invalid = good_prices & ((tensors["high"] < torch.maximum(tensors["open"], tensors["close"])) |
                                 (tensors["low"] > torch.minimum(tensors["open"], tensors["close"])) |
                                 (tensors["low"] > tensors["high"]))
        flags["valid_ohlc"][start:stop] = (good_prices & ~invalid)[offset:].cpu().numpy()
        flags["invalid_ohlc"][start:stop] = invalid[offset:].cpu().numpy()
        flags["zero_amount"][start:stop] = (tensors["amt"][offset:] == 0).cpu().numpy()
        for window in rsi_windows:
            changes = torch.full_like(adjusted, float("nan"))
            changes[1:] = adjusted[1:] - adjusted[:-1]
            gains, losses = changes.clamp(min=0), (-changes).clamp(min=0)
            seed_gain, seed_loss = rolling(gains, window), rolling(losses, window)
            state = rsi_state[window]
            out = torch.full_like(adjusted[offset:], float("nan"))
            for local in range(offset, len(adjusted)):
                good = torch.isfinite(changes[local])
                state["count"] = torch.where(good, state["count"] + 1, 0)
                state["gain"] = torch.where(state["count"] == window, seed_gain[local],
                                             (state["gain"] * (window - 1) + gains[local]) / window)
                state["loss"] = torch.where(state["count"] == window, seed_loss[local],
                                             (state["loss"] * (window - 1) + losses[local]) / window)
                state["gain"] = torch.where(good, state["gain"], 0)
                state["loss"] = torch.where(good, state["loss"], 0)
                score = torch.where(state["loss"] == 0, torch.where(state["gain"] == 0, 50., 100.),
                                    100 - 100 / (1 + state["gain"] / state["loss"]))
                out[local - offset] = torch.where(state["count"] >= window, score, float("nan"))
            values[f"rsi_{window}"][start:stop] = out.cpu().numpy()
        for local in range(offset, len(adjusted)):
            factor = tensors["cumadj"][local]
            known = torch.isfinite(factor) & (factor > 0)
            changed = known & torch.isfinite(previous_factor) & (factor_words[local] != previous_factor_words).any(dim=1)
            small = changed & ((factor / previous_factor - 1).abs() < 0.0001)
            flags["factor_change"][lower + local] = changed.cpu().numpy()
            flags["small_factor_change"][lower + local] = small.cpu().numpy()
            previous_factor = torch.where(known, factor, previous_factor)
            previous_factor_words = torch.where(known[:, None], factor_words[local], previous_factor_words)
        synchronize(device)
        progress(stage="factors", done=stop, total=rows, message=f"{str(device).upper()} 因子与行情检测 {stop} / {rows}")
    if data.arrow_nulls:
        for metric in METRICS:
            if counts[metric]["missing"] != data.arrow_nulls[metric]:
                raise RuntimeError(f"{metric} 的 GPU 缺失统计与 Arrow 元数据存在差异")
    metadata = {"device": str(device), "device_name": device_name(device),
                "torch": torch.__version__, "cuda": torch.version.cuda, "dtype": "float32", "block_days": config.block_days,
                "windows": windows, "cache_key": key, "cache_hit": False, "seconds": round(time.monotonic() - started, 3)}
    profile = {"dates": {"first_loaded": data.dates[0], "last_loaded": data.dates[-1], "start": config.start, "end": config.end,
                         "trading_days": len(data.trading_indices(config))},
               "securities": columns, "stocks": int(data.stock_mask.sum()), "shape": list(data.shape), "fields": counts,
               "diagnostics": {name: int(mask.sum()) for name, mask in flags.items()},
               "fingerprint": data.fingerprint, "source_files": data.source_files}
    if use_cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache.with_name(cache.stem + ".tmp.npz")
        np.savez_compressed(temporary, **{"v_" + k: v for k, v in values.items()}, **{"f_" + k: v for k, v in flags.items()},
                            __metadata__=json.dumps(metadata), __profile__=json.dumps(profile))
        temporary.replace(cache)
    return FeatureBundle(values, flags, metadata, profile)
