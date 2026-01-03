# src/portfolio.py
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd


def compute_weights_from_scores(
    scores: pd.Series,
    returns: Optional[pd.DataFrame] = None,
    vol_window: int = 20,
    top_pct: float = 0.2,
    bottom_pct: float = 0.4,
    long_exposure: float = 0.7,
    short_exposure: float = 0.3,
    max_weight: Optional[float] = None,
    min_names_per_side: int = 3,
    weight_scheme: str = "rank_inv_vol",  # "equal" | "rank" | "inv_vol" | "rank_inv_vol"
) -> pd.Series:
    """
    Transforme un score cross-section en poids long/short.

    - Sélection : TOP_PCT meilleurs / BOTTOM_PCT pires
    - Schémas :
        * equal        : poids égaux
        * rank         : poids ∝ rang (plus extrême => plus gros)
        * inv_vol      : poids ∝ 1/vol (vol calculée sur returns, fenêtre vol_window)
        * rank_inv_vol : poids ∝ rang * (1/vol)
    - Normalisation : somme(longs)=long_exposure ; somme(shorts)=-short_exposure
    - Option : cap max_weight par ligne puis renormalisation
    """
    scores = scores.dropna()
    if scores.empty:
        return pd.Series(dtype=float)

    n = len(scores)
    n_top = max(min_names_per_side, math.floor(n * top_pct))
    n_bottom = max(min_names_per_side, math.floor(n * bottom_pct))

    top = scores.nlargest(n_top).index
    bottom = scores.nsmallest(n_bottom).index

    # rangs percentiles [0,1]
    r = scores.rank(pct=True, ascending=True)

    # inverse-vol
    if returns is None or len(returns) < vol_window:
        inv_vol = pd.Series(1.0, index=scores.index)
    else:
        vol = returns.rolling(vol_window).std().iloc[-1].replace(0.0, np.nan)
        inv_vol = (1.0 / vol).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    if weight_scheme == "equal":
        long_mag = pd.Series(1.0, index=top)
        short_mag = pd.Series(1.0, index=bottom)

    elif weight_scheme == "rank":
        long_mag = r.loc[top]
        short_mag = (1.0 - r.loc[bottom])

    elif weight_scheme == "inv_vol":
        long_mag = inv_vol.loc[top]
        short_mag = inv_vol.loc[bottom]

    elif weight_scheme == "rank_inv_vol":
        long_mag = (r.loc[top] * inv_vol.loc[top]).fillna(0.0)
        short_mag = ((1.0 - r.loc[bottom]) * inv_vol.loc[bottom]).fillna(0.0)

    else:
        raise ValueError(f"Unknown weight_scheme: {weight_scheme}")

    weights = pd.Series(0.0, index=scores.index)

    long_sum = float(long_mag.sum()) if len(long_mag) else 0.0
    short_sum = float(short_mag.sum()) if len(short_mag) else 0.0

    if long_sum > 1e-12:
        weights.loc[long_mag.index] = (long_mag / long_sum) * long_exposure

    if short_sum > 1e-12:
        weights.loc[short_mag.index] = -(short_mag / short_sum) * short_exposure

    # cap par ligne (optionnel)
    if max_weight is not None and max_weight > 0:
        weights = weights.clip(lower=-max_weight, upper=max_weight)

        pos = weights.where(weights > 0, 0.0)
        neg = -weights.where(weights < 0, 0.0)

        pos_sum = float(pos.sum())
        neg_sum = float(neg.sum())

        if pos_sum > 1e-12:
            weights.loc[weights > 0] = pos[pos > 0] * (long_exposure / pos_sum)

        if neg_sum > 1e-12:
            weights.loc[weights < 0] = -neg[weights < 0] * (short_exposure / neg_sum)

    return weights.fillna(0.0)
