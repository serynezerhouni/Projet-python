# src/backtest.py
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .signals import momentum_scores_pocheA
from .portfolio import compute_weights_from_scores


def get_rebalance_dates(prices: pd.DataFrame, freq: str = "M") -> pd.DatetimeIndex:
    """
    Copie fidèle notebook:
    - anchor = dernier jour dispo par période (M => fin de mois)
    - mapping sur les jours de bourse via method="pad"
    """
    idx = prices.index
    if freq == "M":
        anchor = prices.resample("M").last().index
    elif freq == "W-FRI":
        anchor = prices.resample("W-FRI").last().index
    else:
        anchor = prices.resample(freq).last().index

    pos = idx.get_indexer(anchor, method="pad")
    pos = [p for p in pos if p != -1]
    return idx[pos]


def performance_metrics(daily_returns: pd.Series) -> Dict[str, float]:
    """
    Copie fidèle notebook: CAGR, vol ann., Sharpe ann., max drawdown, total return
    """
    r = daily_returns.dropna()
    if r.empty:
        return {}

    cum = (1 + r).cumprod()
    total_days = (r.index[-1] - r.index[0]).days if len(r) > 1 else 0
    years = total_days / 365.25 if total_days > 0 else np.nan

    total_return = float(cum.iloc[-1] - 1.0) if len(cum) > 0 else np.nan
    cagr = float(cum.iloc[-1] ** (1 / years) - 1.0) if years and years > 0 else np.nan

    vol = float(r.std() * math.sqrt(252)) if len(r) > 1 else np.nan
    sharpe = float((r.mean() * 252) / vol) if vol and vol > 1e-9 else np.nan

    dd = cum / cum.cummax() - 1.0
    max_dd = float(dd.min()) if len(dd) > 0 else np.nan

    return {
        "CAGR": cagr,
        "Total Return": total_return,
        "Volatility (ann.)": vol,
        "Sharpe (ann.)": sharpe,
        "Max Drawdown": max_dd,
    }


def backtest_pocheA_momentum(
    prices: pd.DataFrame,
    volumes: pd.DataFrame,
    lookback_days: int = 60,
    top_pct: float = 0.2,
    bottom_pct: float = 0.4,
    vol_scale: bool = False,  # conservé pour compat notebook
    vol_window: int = 20,
    long_exposure: float = 0.7,
    short_exposure: float = 0.3,
    rebal_freq: str = "M",
    transaction_cost_bps: float = 8.0,
    max_weight: Optional[float] = None,
    min_names_per_side: int = 3,
    risk_adjust_by_vol_in_score: bool = True,  # ✅ default notebook
    weight_scheme: str = "rank_inv_vol",
) -> Tuple[pd.Series, pd.DataFrame, Dict[str, float]]:
    """
    Copie fidèle notebook:
    - returns = pct_change().replace(inf).fillna(0)
    - recalcul score à chaque rebal
    - weights via compute_weights_from_scores(...)
    - turnover = sum(abs(w - w_prev))
    - tc = turnover * (bps/1e4), payé le 1er jour après rebal
    """
    prices = prices.dropna(how="all").sort_index()
    volumes = volumes.dropna(how="all").sort_index()

    returns = prices.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    rebal_dates = get_rebalance_dates(prices, rebal_freq)

    weights_history = []
    daily_portfolio_returns = pd.Series(0.0, index=prices.index)
    prev_weights = pd.Series(0.0, index=prices.columns)

    for i, reb_date in enumerate(rebal_dates):
        if reb_date not in prices.index:
            continue
        idx = prices.index.get_loc(reb_date)
        if idx < lookback_days:
            continue

        # 1) score
        scores = momentum_scores_pocheA(
            prices=prices,
            volumes=volumes,
            lookback_days=lookback_days,
            as_of_date=reb_date,
            risk_adjust_by_vol=risk_adjust_by_vol_in_score,
        )
        if scores.empty:
            continue

        # 2) poids
        weights = compute_weights_from_scores(
            scores=scores.reindex(prices.columns),
            returns=returns[prices.columns].loc[:reb_date],
            vol_window=vol_window,
            top_pct=top_pct,
            bottom_pct=bottom_pct,
            long_exposure=long_exposure,
            short_exposure=short_exposure,
            vol_scale=vol_scale,
            max_weight=max_weight,
            min_names_per_side=min_names_per_side,
            weight_scheme=weight_scheme,
        ).reindex(prices.columns).fillna(0.0)

        weights_history.append(pd.DataFrame({"date": reb_date, **weights.to_dict()}, index=[0]))

        # 3) turnover + TC
        turnover = (weights - prev_weights).abs().sum()
        tc = turnover * (transaction_cost_bps / 1e4)

        # 4) appliquer jusqu'au prochain rebal
        start_idx = idx
        end_date = rebal_dates[i + 1] if i + 1 < len(rebal_dates) else prices.index.max()
        end_idx = prices.index.get_loc(end_date)
        period = returns.iloc[start_idx + 1 : end_idx + 1]

        if not period.empty:
            pret = period.dot(weights)
            if turnover > 0:
                pret.iloc[0] -= tc  # coût payé J+1
            daily_portfolio_returns.loc[pret.index] = pret.values

        prev_weights = weights

    weights_df = (
        pd.concat(weights_history, ignore_index=True).set_index("date")
        if weights_history
        else pd.DataFrame(index=pd.DatetimeIndex([]))
    )
    if not weights_df.empty:
        weights_df.index = pd.to_datetime(weights_df.index)

    metrics = performance_metrics(daily_portfolio_returns)
    if not weights_df.empty:
        metrics["Avg Net Exposure"] = float(weights_df.sum(axis=1).mean())
        metrics["Avg Gross Exposure"] = float(weights_df.abs().sum(axis=1).mean())
        metrics["Avg Turnover"] = float(weights_df.diff().abs().sum(axis=1).dropna().mean())

    return daily_portfolio_returns, weights_df, metrics
