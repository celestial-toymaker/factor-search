from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ashare_bt.config import Config
from ashare_bt.data import MarketData, METRICS
from ashare_bt.compute import compute_features
from ashare_bt.status import StatusProvider


@pytest.fixture
def market(tmp_path):
    dates = [d.strftime("%Y-%m-%d") for d in pd.bdate_range("2024-08-01", "2025-01-24") if d.strftime("%Y-%m-%d") != "2025-01-01"]
    securities = ["000001.SZ", "600001.SH", "300001.SZ", "688001.SH", "689009.SH"]
    rng = np.random.default_rng(12)
    returns = rng.normal(.001, .007, (len(dates), len(securities)))
    close = 10 * np.cumprod(1 + returns, axis=0)
    opening = close * .999
    arrays = {"open": opening, "close": close, "high": np.maximum(close, opening) * 1.01,
              "low": np.minimum(close, opening) * .99, "amt": np.full_like(close, 1e9),
              "num_trades": np.full_like(close, 10000), "cumadj": np.ones_like(close)}
    data = MarketData(dates, securities, arrays)
    config = Config(root=tmp_path, start="2025-01-02", end="2025-01-24", initial_cash="1000000",
                    device="reference", max_positions=3, auto_sync_status=False)
    features = compute_features(data, config, use_cache=False)
    status = StatusProvider.synthetic(dates, securities)
    return config, data, features, status
