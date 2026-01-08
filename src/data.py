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
    Télécharge prix ajustés (Close auto_adjust) + volumes via yfinance.
    Retourne (prices, volumes) au format DataFrame (dates en index, tickers en colonnes).
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
        raise ValueError("Téléchargement vide : vérifier tickers/période.")

    # yfinance renvoie souvent un MultiIndex (Open/High/Low/Close/Volume, ticker)
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"].copy()
        volume = raw["Volume"].copy()
        close.columns = close.columns.get_level_values(-1)
        volume.columns = volume.columns.get_level_values(-1)
    else:
        # cas 1 ticker
        close = raw[["Close"]].copy()
        volume = raw[["Volume"]].copy()
        close.columns = [tickers[0]]
        volume.columns = [tickers[0]]

    close = close.dropna(how="all").sort_index()
    volume = volume.dropna(how="all").sort_index()

  
    for t in tickers:
        if t not in close.columns:
            close[t] = np.nan
        if t not in volume.columns:
            volume[t] = np.nan

    close = close[tickers]
    volume = volume[tickers]
    return close, volume
