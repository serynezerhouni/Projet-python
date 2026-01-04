# src/analysis.py
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
import statsmodels.api as sm
from pandas_datareader import data as web

from .backtest import backtest_pocheA_momentum


# =========================================
# Helpers marché / CAPM
# =========================================

def get_market_returns(start_date: str, end_date: str, ticker: str = "^GSPC") -> pd.Series:
    """
    Rendements quotidiens de l'indice de marché (SP500 par défaut).
    """
    data = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
    r_mkt = data["Close"].pct_change().dropna()
    r_mkt.name = "MKT"
    return r_mkt


def run_capm(
    portfolio_returns: pd.Series,
    market_returns: pd.Series,
    rf_annual: float = 0.0,
) -> Tuple[Dict[str, float], sm.regression.linear_model.RegressionResultsWrapper]:
    """
    CAPM : (Rp - rf) = alpha + beta (Rm - rf) + epsilon
    """
    df = pd.concat([portfolio_returns, market_returns], axis=1, join="inner").dropna()
    df.columns = ["PORT", "MKT"]

    rf_daily = rf_annual / 252.0
    excess_port = df["PORT"] - rf_daily
    excess_mkt = df["MKT"] - rf_daily

    X = sm.add_constant(excess_mkt)
    model = sm.OLS(excess_port, X).fit()

    alpha_daily = float(model.params["const"])
    beta = float(model.params["MKT"])
    alpha_ann = (1.0 + alpha_daily) ** 252 - 1.0

    stats = {
        "alpha_ann": alpha_ann,
        "beta": beta,
        "r2": float(model.rsquared),
        "t_alpha": float(model.tvalues["const"]),
        "t_beta": float(model.tvalues["MKT"]),
    }
    return stats, model


def compare_exposures_capm(
    prices: pd.DataFrame,
    volumes: pd.DataFrame,
    top_pct: float = 0.2,
    bottom_pct: float = 0.4,
    vol_window: int = 20,
    rebal_freq: str = "M",
    transaction_cost_bps: float = 8.0,
    max_weight: Optional[float] = None,
    min_names_per_side: int = 3,
    weight_scheme: str = "rank_inv_vol",
    risk_adjust_by_vol_in_score: bool = False,
) -> pd.DataFrame:
    """
    Compare 50/50 vs 70/30 (mêmes paramètres sinon), + CAPM vs SP500.
    """
    rows = []

    for (L, S) in [(0.5, 0.5), (0.7, 0.3)]:
        ret, w_df, met = backtest_pocheA_momentum(
            prices=prices,
            volumes=volumes,
            lookback_days=60,
            top_pct=top_pct,
            bottom_pct=bottom_pct,
            vol_window=vol_window,
            long_exposure=L,
            short_exposure=S,
            rebal_freq=rebal_freq,
            transaction_cost_bps=transaction_cost_bps,
            max_weight=max_weight,
            min_names_per_side=min_names_per_side,
            risk_adjust_by_vol_in_score=risk_adjust_by_vol_in_score,
            weight_scheme=weight_scheme,
        )

        start = ret.index.min().date().isoformat()
        end = ret.index.max().date().isoformat()
        mkt = get_market_returns(start, end)

        capm_stats, _ = run_capm(ret, mkt)

        rows.append({
            "L": L,
            "S": S,
            "CAGR": met.get("CAGR", np.nan),
            "Sharpe": met.get("Sharpe (ann.)", np.nan),
            "Vol": met.get("Volatility (ann.)", np.nan),
            "MaxDD": met.get("Max Drawdown", np.nan),
            "Avg Turnover": met.get("Avg Turnover", np.nan), 
            "beta_mkt": capm_stats["beta"],
            "alpha_ann": capm_stats["alpha_ann"],
            "R2_CAPM": capm_stats["r2"],
        })

    return pd.DataFrame(rows)


# =========================================
# Fama-French 3 facteurs
# =========================================

def get_ff3_factors(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Facteurs Fama-French 3 + RF (daily). Les données sont en % -> converties en décimal.
    """
    ff = web.DataReader("F-F_Research_Data_Factors_Daily", "famafrench", start_date, end_date)[0]
    ff = ff / 100.0
    ff.index = ff.index.tz_localize(None)
    return ff[["Mkt-RF", "SMB", "HML", "RF"]]


def run_ff3(
    portfolio_returns: pd.Series,
    ff: pd.DataFrame
) -> Tuple[Dict[str, float], sm.regression.linear_model.RegressionResultsWrapper]:
    """
    (Rp - RF) = alpha + b_m(Mkt-RF) + b_s(SMB) + b_h(HML) + eps
    """
    df = pd.concat([portfolio_returns, ff], axis=1, join="inner").dropna()
    df.columns = ["PORT", "Mkt-RF", "SMB", "HML", "RF"]

    y = df["PORT"] - df["RF"]
    X = sm.add_constant(df[["Mkt-RF", "SMB", "HML"]])

    model = sm.OLS(y, X).fit()

    alpha_daily = float(model.params["const"])
    alpha_ann = (1.0 + alpha_daily) ** 252 - 1.0

    stats = {
        "alpha_ann": alpha_ann,
        "r2": float(model.rsquared),
        "load_Mkt": float(model.params["Mkt-RF"]),
        "load_SMB": float(model.params["SMB"]),
        "load_HML": float(model.params["HML"]),
    }
    return stats, model


def analyse_ff3_sur_strategie(
    prices: pd.DataFrame,
    volumes: pd.DataFrame,
    long_expo: float = 0.7,
    short_expo: float = 0.3,
    top_pct: float = 0.2,
    bottom_pct: float = 0.4,
    vol_window: int = 20,
    rebal_freq: str = "M",
    transaction_cost_bps: float = 8.0,
    max_weight: Optional[float] = None,
    min_names_per_side: int = 3,
    weight_scheme: str = "rank_inv_vol",
    risk_adjust_by_vol_in_score: bool = False,
) -> Tuple[pd.Series, Dict[str, float], Dict[str, float]]:
    """
    Run stratégie puis FF3.
    """
    ret, w_df, metrics = backtest_pocheA_momentum(
        prices=prices,
        volumes=volumes,
        lookback_days=60,
        top_pct=top_pct,
        bottom_pct=bottom_pct,
        vol_window=vol_window,
        long_exposure=long_expo,
        short_exposure=short_expo,
        rebal_freq=rebal_freq,
        transaction_cost_bps=transaction_cost_bps,
        max_weight=max_weight,
        min_names_per_side=min_names_per_side,
        risk_adjust_by_vol_in_score=risk_adjust_by_vol_in_score,
        weight_scheme=weight_scheme,
    )

    start = ret.index.min().date().isoformat()
    end = ret.index.max().date().isoformat()
    ff = get_ff3_factors(start, end)
    ff_stats, _ = run_ff3(ret, ff)
    return ret, metrics, ff_stats


# =========================================
# Analyse des sous-périodes
# =========================================

def run_subperiods_table(
    prices: pd.DataFrame,
    volumes: pd.DataFrame,
    periods: Optional[List[Tuple[str, str]]] = None,
    long_expo: float = 0.7,
    short_expo: float = 0.3,
    top_pct: float = 0.2,
    bottom_pct: float = 0.4,
    vol_window: int = 20,
    rebal_freq: str = "M",
    transaction_cost_bps: float = 8.0,
    max_weight: Optional[float] = None,
    min_names_per_side: int = 3,
    weight_scheme: str = "rank_inv_vol",
    risk_adjust_by_vol_in_score: bool = False,
) -> pd.DataFrame:
    if periods is None:
        periods = [
            ("2010-01-01", "2014-12-31"),
            ("2015-01-01", "2019-12-31"),
            ("2020-01-01", "2024-12-31"),
        ]

    rows = []
    for (start, end) in periods:
        p = prices.loc[start:end].copy()
        v = volumes.loc[start:end].copy()

        ret, w_df, m = backtest_pocheA_momentum(
            prices=p,
            volumes=v,
            lookback_days=60,
            top_pct=top_pct,
            bottom_pct=bottom_pct,
            vol_window=vol_window,
            long_exposure=long_expo,
            short_exposure=short_expo,
            rebal_freq=rebal_freq,
            transaction_cost_bps=transaction_cost_bps,
            max_weight=max_weight,
            min_names_per_side=min_names_per_side,
            risk_adjust_by_vol_in_score=risk_adjust_by_vol_in_score,
            weight_scheme=weight_scheme,
        )

        rows.append({
            "start": start,
            "end": end,
            "CAGR": m.get("CAGR", np.nan),
            "Sharpe": m.get("Sharpe (ann.)", np.nan),
            "Vol": m.get("Volatility (ann.)", np.nan),
            "MaxDD": m.get("Max Drawdown", np.nan),
        })

    return pd.DataFrame(rows)


# =========================================
# Sensibilité hyperparamètres
# =========================================

def hyperparam_grid_search(
    prices: pd.DataFrame,
    volumes: pd.DataFrame,
    top_list=(0.1, 0.2),
    bottom_list=(0.2, 0.4),
    exposure_list=((0.5, 0.5), (0.7, 0.3)),
    vol_window: int = 20,
    rebal_freq: str = "M",
    transaction_cost_bps: float = 8.0,
    max_weight: Optional[float] = None,
    min_names_per_side: int = 3,
    weight_scheme: str = "rank_inv_vol",
    risk_adjust_by_vol_in_score: bool = False,
) -> pd.DataFrame:
    rows = []
    for top in top_list:
        for bottom in bottom_list:
            for (L, S) in exposure_list:
                ret, w_df, m = backtest_pocheA_momentum(
                    prices=prices,
                    volumes=volumes,
                    lookback_days=60,
                    top_pct=top,
                    bottom_pct=bottom,
                    vol_window=vol_window,
                    long_exposure=L,
                    short_exposure=S,
                    rebal_freq=rebal_freq,
                    transaction_cost_bps=transaction_cost_bps,
                    max_weight=max_weight,
                    min_names_per_side=min_names_per_side,
                    risk_adjust_by_vol_in_score=risk_adjust_by_vol_in_score,
                    weight_scheme=weight_scheme,
                )
                rows.append({
                    "TOP_PCT": top,
                    "BOTTOM_PCT": bottom,
                    "L": L,
                    "S": S,
                    "CAGR": m.get("CAGR", np.nan),
                    "Sharpe": m.get("Sharpe (ann.)", np.nan),
                    "Vol": m.get("Volatility (ann.)", np.nan),
                    "MaxDD": m.get("Max Drawdown", np.nan),
                })
    return pd.DataFrame(rows)


# =========================================
# Ablation
# =========================================

def run_ablation_tests(
    prices: pd.DataFrame,
    volumes: pd.DataFrame,
    top_pct: float = 0.2,
    bottom_pct: float = 0.4,
    long_exposure: float = 0.7,
    short_exposure: float = 0.3,
    vol_window: int = 20,
    rebal_freq: str = "M",
    transaction_cost_bps: float = 8.0,
    max_weight: Optional[float] = None,
    min_names_per_side: int = 3,
    weight_scheme: str = "rank_inv_vol",
    risk_adjust_by_vol_in_score: bool = False,
) -> pd.DataFrame:
    """
    Ablation alignée sur le produit final :
    - poids = rank_inv_vol
    - expo = 70/30
    - top/bottom = 0.2/0.4
    - TC = 8 bps
    - rebal = mensuel
    - score NON risk-adjust (risk_adjust_by_vol_in_score=False)
    Variantes : baseline, no_RSI, no_volume, no_RSI_no_volume
    """
    configs = [
        ("baseline_final", True, True),
        ("no_RSI", False, True),
        ("no_volume", True, False),
        ("no_RSI_no_volume", False, False),
    ]

    rows = []
    for name, use_rsi, use_vol in configs:
        ret, w_df, m = backtest_pocheA_momentum(
            prices=prices,
            volumes=volumes,
            lookback_days=60,
            top_pct=top_pct,
            bottom_pct=bottom_pct,
            vol_window=vol_window,
            long_exposure=long_exposure,
            short_exposure=short_exposure,
            rebal_freq=rebal_freq,
            transaction_cost_bps=transaction_cost_bps,
            max_weight=max_weight,
            min_names_per_side=min_names_per_side,
            risk_adjust_by_vol_in_score=risk_adjust_by_vol_in_score,
            weight_scheme=weight_scheme,
            use_rsi=use_rsi,
            use_volume_penalty=use_vol,
        )

        rows.append({
            "variant": name,
            "CAGR": m.get("CAGR", np.nan),
            "Sharpe": m.get("Sharpe (ann.)", np.nan),
            "Vol": m.get("Volatility (ann.)", np.nan),
            "MaxDD": m.get("Max Drawdown", np.nan),
        })

    return pd.DataFrame(rows)
