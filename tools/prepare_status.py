from pathlib import Path
from ashare_bt.config import Config
from ashare_bt.data import load_data
from ashare_bt.status import ensure_status
from ashare_bt.utils import atomic_json


def progress(**message):
    if message.get("stage") != "status" or message.get("done", 0) % 10 == 0:
        print(message, flush=True)


if __name__ == "__main__":
    config = Config()
    data = load_data(config, progress)
    statuses = ensure_status(data, config, progress)
    import numpy as np
    required = np.isfinite(data.arrays["close"]) & (data.arrays["amt"] > 0) & data.stock_mask[None, :]
    required[:data.date_index[config.start] - 1] = False
    report = statuses.coverage(required)
    atomic_json(config.root / "data" / "status" / "coverage.json", report)
    print(report, flush=True)
