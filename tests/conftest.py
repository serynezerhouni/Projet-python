"""Synthetic market data shared by the tests (no download needed)."""
import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def market():
    """700 business days of prices and volumes for 20 random stocks (fixed seed)."""
    rng = np.random.default_rng(0)
    n_days, n_stocks = 700, 20
    index = pd.bdate_range("2015-01-01", periods=n_days)
    columns = [f"S{i:02d}" for i in range(n_stocks)]

    drift = rng.normal(0.0003, 0.0004, n_stocks)
    vol = rng.uniform(0.01, 0.03, n_stocks)
    log_returns = rng.normal(drift, vol, size=(n_days, n_stocks))

    prices = pd.DataFrame(100.0 * np.exp(np.cumsum(log_returns, axis=0)), index=index, columns=columns)
    volumes = pd.DataFrame(rng.uniform(2e5, 3e6, size=(n_days, n_stocks)), index=index, columns=columns)
    return prices, volumes
