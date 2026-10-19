import numpy as np
import pandas as pd
import pytest

from src.risk import (
    christoffersen_independence,
    historical_es,
    historical_var,
    kupiec_pof,
    parametric_var,
    rolling_var,
    var_backtest,
)


@pytest.fixture
def normal_returns():
    rng = np.random.default_rng(42)
    return pd.Series(rng.normal(0.0, 0.01, 200_000))


def test_var_matches_the_normal_quantile(normal_returns):
    # 99% one-day VaR of N(0, 1%) is 2.326 * 1% = 2.33%
    assert historical_var(normal_returns, 0.99) == pytest.approx(0.02326, rel=0.02)
    assert parametric_var(normal_returns, 0.99) == pytest.approx(0.02326, rel=0.02)
    assert historical_es(normal_returns, 0.99) > historical_var(normal_returns, 0.99)


def test_kupiec_accepts_right_coverage_and_rejects_too_many_exceptions():
    _, p_right = kupiec_pof(n_obs=1000, n_exceptions=10, alpha=0.99)
    _, p_too_many = kupiec_pof(n_obs=1000, n_exceptions=30, alpha=0.99)

    assert p_right > 0.9
    assert p_too_many < 0.001


def test_christoffersen_flags_clustered_exceptions():
    clustered = pd.Series([0] * 480 + [1] * 10 + [0] * 510)
    isolated = pd.Series(([1] + [0] * 99) * 10)

    _, p_clustered = christoffersen_independence(clustered)
    _, p_isolated = christoffersen_independence(isolated)

    assert p_clustered < 0.01
    assert p_isolated > 0.05


def test_var_forecast_does_not_see_the_same_day_return():
    rng = np.random.default_rng(3)
    r = pd.Series(rng.normal(0.0, 0.01, 600), index=pd.bdate_range("2020-01-01", periods=600))
    shocked = r.copy()
    shocked.iloc[400] = -0.5

    var = rolling_var(r, alpha=0.99, window=250)
    var_shocked = rolling_var(shocked, alpha=0.99, window=250)

    assert var.iloc[400] == var_shocked.iloc[400]
    assert var_shocked.iloc[401] > var.iloc[401]


def test_var_backtest_on_well_specified_model():
    rng = np.random.default_rng(11)
    r = pd.Series(rng.normal(0.0, 0.01, 3000), index=pd.bdate_range("2010-01-01", periods=3000))

    summary, daily = var_backtest(r, alpha=0.99, window=250)

    assert summary["observations"] == len(daily)
    assert summary["kupiec_pvalue"] > 0.01
    assert summary["christoffersen_pvalue"] > 0.01
