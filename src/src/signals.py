# src/signals.py
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def momentum_scores_pocheA(
    prices: pd.DataFrame,
    volumes: pd.DataFrame,
    lookback_days: int,
    as_of_date: Optional[pd.Timestamp] = None,
    roc_window: int = 126,
    ma_short: int = 50,
    ma_long: int = 200,
    vol_sigma_window: int = 60,
    rsi_window: int = 14,
    w_roc: float = 0.7,
    w_ma: float = 0.3,
    rsi_penalty_level: float = 80.0,
    rsi_penalty_factor: float = 0.5,
    use_rsi: bool = True,
    use_volume_penalty: bool = True,
    volume_threshold: float = 500_000.0,
    volume_penalty_factor: float = 0.5,
    risk_adjust_by_vol: bool = False,
) -> pd.Series:
    """
    Score momentum composite (Poche A) :
    - ROC 6 mois (roc_window)
    - MA50/MA200 (tendance)
    - Rang cross-section + combinaison (w_roc, w_ma)
    - Pénalité RSI (optionnelle)
    - Pénalité volume (optionnelle)
    - Option : division par la vol dans le score (risk_adjust_by_vol)

    IMPORTANT (produit final):
    - on met généralement risk_adjust_by_vol=False
      car le low-vol est géré dans les poids (rank * inverse-vol).
    """
    prices = prices.dropna(how="all").sort_index()
    volumes = volumes.dropna(how="all").sort_index()

    if as_of_date is None:
        as_of_date = prices.index.max()
    if as_of_date not in prices.index:
        # prend la dernière date dispo avant as_of_date
        as_of_date = prices.index[prices.index.get_loc(as_of_date, method="pad")]

    idx_pos = prices.index.get_loc(as_of_date)
    required = max(ma_long, roc_window, vol_sigma_window, rsi_window)

    # fallback si pas assez d'historique
    if idx_pos < required:
        if idx_pos < lookback_days:
            return pd.Series(dtype=float)
        current = prices.iloc[idx_pos]
        past = prices.shift(lookback_days).iloc[idx_pos]
        return (current / past - 1.0).replace([np.inf, -np.inf], np.nan).dropna()

    hist_prices = prices.iloc[: idx_pos + 1]
    hist_volumes = volumes.iloc[: idx_pos + 1]

    # ROC
    price_now = hist_prices.iloc[-1]
    price_roc = hist_prices.shift(roc_window).iloc[-1]
    roc = (price_now / price_roc) - 1.0

    # MA50 / MA200
    ma_s = hist_prices.rolling(ma_short).mean().iloc[-1]
    ma_l = hist_prices.rolling(ma_long).mean().iloc[-1]
    ma_ratio = (ma_s / ma_l)

    # volatilité pour risk-adjust éventuel
    returns = hist_prices.pct_change(fill_method=None)
    vol = returns.rolling(vol_sigma_window).std().iloc[-1]

    # RSI
    delta = hist_prices.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.rolling(rsi_window).mean().iloc[-1]
    avg_loss = loss.rolling(rsi_window).mean().iloc[-1]
    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # volume moyen
    avg_volume = hist_volumes.rolling(60).mean().iloc[-1]

    # nettoyage
    roc = roc.replace([np.inf, -np.inf], np.nan)
    ma_ratio = ma_ratio.replace([np.inf, -np.inf], np.nan)
    vol = vol.replace([np.inf, -np.inf, 0.0], np.nan)
    rsi = rsi.replace([np.inf, -np.inf], np.nan)
    avg_volume = avg_volume.replace([np.inf, -np.inf], np.nan)

    # rangs cross-section
    rank_roc = roc.rank(pct=True, ascending=True)
    rank_ma = ma_ratio.rank(pct=True, ascending=True)
    score_base = w_roc * rank_roc + w_ma * rank_ma

    score_adj = score_base / (vol + 1e-8) if risk_adjust_by_vol else score_base

    penalty = pd.Series(1.0, index=score_adj.index)

    if use_rsi:
        penalty[rsi > rsi_penalty_level] = rsi_penalty_factor

    if use_volume_penalty:
        penalty[(avg_volume < volume_threshold).fillna(False)] *= volume_penalty_factor

    score_final = (score_adj * penalty).replace([np.inf, -np.inf], np.nan).dropna()
    return score_final
