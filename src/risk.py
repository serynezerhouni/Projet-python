# src/risk.py
"""
Market risk toolkit for the strategy's daily P&L.

All VaR / ES figures are returned as POSITIVE numbers expressed as a fraction
of portfolio value (e.g. 0.012 = a 1.2% one-day loss).

Contents
- Historical, parametric (Gaussian) and Cornish-Fisher VaR
- Historical and parametric Expected Shortfall
- Stressed VaR: historical VaR on the worst 12-month window (Basel 2.5 style)
- Rolling VaR forecasts and VaR backtesting (Kupiec POF, Christoffersen
  independence, Basel traffic light)
- Historical stress scenarios replayed on the strategy's own returns
- Rolling volatility and rolling market beta
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import xlogy

TRADING_DAYS = 252


# =========================
# Helpers
# =========================
def clean_returns(returns: pd.Series) -> pd.Series:
    """
    Drop NaNs and the leading flat period before the first rebalance
    (the backtest returns 0.0 until the strategy starts trading).
    """
    r = pd.Series(returns).astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    nonzero = r.ne(0.0)
    if nonzero.any():
        r = r.loc[nonzero.idxmax():]
    return r


# =========================
# VaR / ES (one-day, single sample)
# =========================
def historical_var(returns: pd.Series, alpha: float = 0.99) -> float:
    """Historical VaR: empirical (1 - alpha) quantile of returns, sign flipped."""
    r = pd.Series(returns).dropna()
    return float(-np.quantile(r, 1.0 - alpha))


def historical_es(returns: pd.Series, alpha: float = 0.99) -> float:
    """Historical Expected Shortfall: average loss beyond the historical VaR."""
    r = pd.Series(returns).dropna()
    q = np.quantile(r, 1.0 - alpha)
    tail = r[r <= q]
    return float(-tail.mean())


def parametric_var(returns: pd.Series, alpha: float = 0.99) -> float:
    """Gaussian (variance-covariance) VaR on the portfolio return series."""
    r = pd.Series(returns).dropna()
    mu, sigma = r.mean(), r.std(ddof=1)
    z = stats.norm.ppf(1.0 - alpha)
    return float(-(mu + z * sigma))


def parametric_es(returns: pd.Series, alpha: float = 0.99) -> float:
    """Gaussian Expected Shortfall."""
    r = pd.Series(returns).dropna()
    mu, sigma = r.mean(), r.std(ddof=1)
    z = stats.norm.ppf(1.0 - alpha)
    return float(-(mu - sigma * stats.norm.pdf(z) / (1.0 - alpha)))


def cornish_fisher_var(returns: pd.Series, alpha: float = 0.99) -> float:
    """
    Modified VaR (Cornish-Fisher expansion): adjusts the Gaussian quantile
    for the skewness and excess kurtosis of the return distribution.
    """
    r = pd.Series(returns).dropna()
    mu, sigma = r.mean(), r.std(ddof=1)
    s = stats.skew(r)
    k = stats.kurtosis(r)  # excess kurtosis
    z = stats.norm.ppf(1.0 - alpha)
    z_cf = (
        z
        + (z**2 - 1.0) * s / 6.0
        + (z**3 - 3.0 * z) * k / 24.0
        - (2.0 * z**3 - 5.0 * z) * s**2 / 36.0
    )
    return float(-(mu + z_cf * sigma))


def scale_to_horizon(var_1d: float, horizon_days: int = 10) -> float:
    """Square-root-of-time scaling (regulatory 10-day convention)."""
    return float(var_1d * np.sqrt(horizon_days))


# =========================
# Stressed VaR
# =========================
def stressed_var(
    returns: pd.Series,
    alpha: float = 0.99,
    window: int = TRADING_DAYS,
) -> Dict[str, object]:
    """
    Stressed VaR (Basel 2.5 logic): historical VaR computed on the continuous
    12-month window that produces the highest VaR in the sample.

    Returns the stressed VaR, the ES on the same window and the window dates.
    """
    r = clean_returns(returns)
    if len(r) < window:
        raise ValueError("Not enough observations for the stressed window.")

    rolling_q = r.rolling(window).quantile(1.0 - alpha)
    end_pos = int(np.nanargmin(rolling_q.values))  # most negative quantile = worst window
    start_pos = end_pos - window + 1
    stress_slice = r.iloc[start_pos : end_pos + 1]

    return {
        "stressed_var": historical_var(stress_slice, alpha),
        "stressed_es": historical_es(stress_slice, alpha),
        "window_start": stress_slice.index[0].date().isoformat(),
        "window_end": stress_slice.index[-1].date().isoformat(),
    }


# =========================
# Rolling VaR + backtesting
# =========================
def rolling_var(
    returns: pd.Series,
    alpha: float = 0.99,
    window: int = TRADING_DAYS,
    method: str = "historical",
) -> pd.Series:
    """
    One-day-ahead VaR forecast: the VaR for day t only uses returns up to t-1
    (no look-ahead). Returned as a positive number aligned on day t.
    """
    r = clean_returns(returns)
    if method == "historical":
        var = -r.rolling(window).quantile(1.0 - alpha)
    elif method == "parametric":
        z = stats.norm.ppf(1.0 - alpha)
        var = -(r.rolling(window).mean() + z * r.rolling(window).std())
    else:
        raise ValueError(f"Unknown method: {method}")
    return var.shift(1).rename(f"VaR_{int(alpha * 100)}_{method}")


def kupiec_pof(n_obs: int, n_exceptions: int, alpha: float = 0.99) -> Tuple[float, float]:
    """
    Kupiec proportion-of-failures test.
    H0: the exception rate equals 1 - alpha. Returns (LR statistic, p-value).
    """
    p = 1.0 - alpha
    x, n = n_exceptions, n_obs
    if n == 0:
        return np.nan, np.nan
    phat = x / n
    log_l0 = xlogy(n - x, 1.0 - p) + xlogy(x, p)
    log_l1 = xlogy(n - x, 1.0 - phat) + xlogy(x, phat)
    lr = -2.0 * (log_l0 - log_l1)
    return float(lr), float(1.0 - stats.chi2.cdf(lr, df=1))


def christoffersen_independence(exceptions: pd.Series) -> Tuple[float, float]:
    """
    Christoffersen independence test: are exceptions clustered in time?
    H0: an exception today does not depend on an exception yesterday.
    Returns (LR statistic, p-value).
    """
    e = exceptions.astype(int).values
    if len(e) < 2:
        return np.nan, np.nan
    prev, curr = e[:-1], e[1:]
    n00 = int(np.sum((prev == 0) & (curr == 0)))
    n01 = int(np.sum((prev == 0) & (curr == 1)))
    n10 = int(np.sum((prev == 1) & (curr == 0)))
    n11 = int(np.sum((prev == 1) & (curr == 1)))

    pi0 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0.0
    pi1 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0.0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)

    log_l0 = xlogy(n00 + n10, 1.0 - pi) + xlogy(n01 + n11, pi)
    log_l1 = xlogy(n00, 1.0 - pi0) + xlogy(n01, pi0) + xlogy(n10, 1.0 - pi1) + xlogy(n11, pi1)
    lr = -2.0 * (log_l0 - log_l1)
    return float(lr), float(1.0 - stats.chi2.cdf(lr, df=1))


def basel_traffic_light(n_exceptions_250d: int) -> str:
    """Basel zones for a 99% VaR backtested on the last 250 trading days."""
    if n_exceptions_250d <= 4:
        return "green"
    if n_exceptions_250d <= 9:
        return "yellow"
    return "red"


def var_backtest(
    returns: pd.Series,
    alpha: float = 0.99,
    window: int = TRADING_DAYS,
    method: str = "historical",
) -> Tuple[Dict[str, object], pd.DataFrame]:
    """
    Backtests the rolling one-day VaR against realised returns.
    Returns a summary dict and a daily DataFrame (return, VaR, exception flag).
    """
    r = clean_returns(returns)
    var = rolling_var(r, alpha=alpha, window=window, method=method)
    df = pd.concat([r.rename("return"), var.rename("VaR")], axis=1).dropna()
    df["exception"] = df["return"] < -df["VaR"]

    n = len(df)
    x = int(df["exception"].sum())
    lr_pof, p_pof = kupiec_pof(n, x, alpha)
    lr_ind, p_ind = christoffersen_independence(df["exception"])
    last_250 = int(df["exception"].iloc[-TRADING_DAYS:].sum())

    summary = {
        "confidence": alpha,
        "method": method,
        "observations": n,
        "exceptions": x,
        "expected_exceptions": round(n * (1.0 - alpha), 1),
        "exception_rate": x / n if n else np.nan,
        "kupiec_LR": lr_pof,
        "kupiec_pvalue": p_pof,
        "christoffersen_LR": lr_ind,
        "christoffersen_pvalue": p_ind,
        "exceptions_last_250d": last_250,
        "basel_zone_last_250d": basel_traffic_light(last_250) if alpha == 0.99 else "n/a",
    }
    return summary, df


# =========================
# Historical stress scenarios
# =========================
DEFAULT_STRESS_SCENARIOS: Dict[str, Tuple[str, str]] = {
    "US downgrade (Jul-Oct 2011)": ("2011-07-22", "2011-10-03"),
    "China / oil selloff (Aug 2015-Feb 2016)": ("2015-08-10", "2016-02-11"),
    "Q4 2018 selloff": ("2018-09-20", "2018-12-24"),
    "COVID crash (Feb-Mar 2020)": ("2020-02-19", "2020-03-23"),
    "Rate shock (2022)": ("2022-01-03", "2022-10-12"),
    "Tariff shock (Feb-Apr 2025)": ("2025-02-19", "2025-04-08"),
}


def stress_scenarios(
    returns: pd.Series,
    scenarios: Optional[Dict[str, Tuple[str, str]]] = None,
) -> pd.DataFrame:
    """
    Replays historical market stress windows on the strategy's own returns:
    cumulative P&L, worst day and max drawdown inside each window.
    """
    r = clean_returns(returns)
    scenarios = scenarios or DEFAULT_STRESS_SCENARIOS
    rows = []
    for name, (start, end) in scenarios.items():
        w = r.loc[start:end]
        if w.empty:
            continue
        equity = (1.0 + w).cumprod()
        rows.append({
            "scenario": name,
            "start": start,
            "end": end,
            "cumulative_return": float(equity.iloc[-1] - 1.0),
            "worst_day": float(w.min()),
            "max_drawdown": float((equity / equity.cummax() - 1.0).min()),
        })
    return pd.DataFrame(rows).set_index("scenario") if rows else pd.DataFrame()


# =========================
# Rolling volatility / beta
# =========================
def rolling_volatility(returns: pd.Series, window: int = 63) -> pd.Series:
    """Annualised rolling volatility (63 days ~ one quarter)."""
    r = clean_returns(returns)
    return (r.rolling(window).std() * np.sqrt(TRADING_DAYS)).rename(f"vol_{window}d")


def rolling_beta(
    portfolio_returns: pd.Series,
    market_returns: pd.Series,
    window: int = 126,
) -> pd.Series:
    """Rolling OLS beta of the strategy on the market (126 days ~ six months)."""
    p = clean_returns(portfolio_returns)
    m = market_returns.squeeze() if isinstance(market_returns, pd.DataFrame) else pd.Series(market_returns)
    df = pd.concat([p.rename("p"), m.rename("m")], axis=1, join="inner").dropna()
    cov = df["p"].rolling(window).cov(df["m"])
    var = df["m"].rolling(window).var()
    return (cov / var).rename(f"beta_{window}d")


# =========================
# One-call summary
# =========================
def var_summary(
    returns: pd.Series,
    alphas: Iterable[float] = (0.95, 0.99),
    horizon_days: int = 10,
) -> pd.DataFrame:
    """VaR / ES by method and confidence level, one-day and scaled to horizon_days."""
    r = clean_returns(returns)
    rows = []
    for a in alphas:
        h_var = historical_var(r, a)
        rows.append({
            "confidence": a,
            "hist_VaR_1d": h_var,
            "param_VaR_1d": parametric_var(r, a),
            "cornish_fisher_VaR_1d": cornish_fisher_var(r, a),
            "hist_ES_1d": historical_es(r, a),
            "param_ES_1d": parametric_es(r, a),
            f"hist_VaR_{horizon_days}d": scale_to_horizon(h_var, horizon_days),
        })
    return pd.DataFrame(rows).set_index("confidence")
