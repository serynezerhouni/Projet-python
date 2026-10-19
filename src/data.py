# src/data.py
from __future__ import annotations

import datetime as dt
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf


def get_prices_and_volume(
    tickers: List[str],
    start: str,
    end: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Download adjusted close prices (auto_adjust) and volumes from yfinance.
    Returns (prices, volumes): dates in the index, tickers in the columns.
    """
    if end is None:
        end = dt.date.today().isoformat()

    raw = yf.download(
        " ".join(tickers),
        start=start,
        end=end,
        progress=False,
        auto_adjust=True,
    )
    if raw.empty:
        raise ValueError("Empty download: check the tickers and the dates.")

    # yfinance usually returns a MultiIndex (Open/High/Low/Close/Volume, ticker)
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"].copy()
        volume = raw["Volume"].copy()
        close.columns = close.columns.get_level_values(-1)
        volume.columns = volume.columns.get_level_values(-1)
    else:
        # single ticker
        close = raw[["Close"]].copy()
        volume = raw[["Volume"]].copy()
        close.columns = [tickers[0]]
        volume.columns = [tickers[0]]

    close = close.dropna(how="all").sort_index()
    volume = volume.dropna(how="all").sort_index()

    # keep every requested ticker, even if the download returned nothing for it
    for t in tickers:
        if t not in close.columns:
            close[t] = np.nan
        if t not in volume.columns:
            volume[t] = np.nan

    close = close[tickers]
    volume = volume[tickers]
    return close, volume
