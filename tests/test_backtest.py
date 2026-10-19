import numpy as np
import pandas as pd
import pytest

from src.backtest import backtest_momentum, performance_metrics


def test_weights_ignore_future_prices(market):
    prices, volumes = market
    cutoff = prices.index[450]

    rng = np.random.default_rng(1)
    shocked = prices.copy()
    future = shocked.index > cutoff
    shocked.loc[future] = shocked.loc[future] * rng.uniform(0.5, 2.0, size=shocked.shape[1])

    _, w_base, _ = backtest_momentum(prices, volumes)
    _, w_shocked, _ = backtest_momentum(shocked, volumes)

    pd.testing.assert_frame_equal(w_base.loc[:cutoff], w_shocked.loc[:cutoff])


def test_returns_start_the_day_after_the_first_rebalance(market):
    prices, volumes = market
    ret, weights, _ = backtest_momentum(prices, volumes)
    first = weights.index[0]

    assert (ret.loc[:first] == 0.0).all()
    assert ret.loc[ret.index > first].iloc[0] != 0.0


def test_costs_are_charged_on_turnover_on_the_first_day_of_each_period(market):
    prices, volumes = market
    ret_free, weights, _ = backtest_momentum(prices, volumes, transaction_cost_bps=0.0)
    ret_cost, _, _ = backtest_momentum(prices, volumes, transaction_cost_bps=100.0)
    cost = ret_free - ret_cost

    turnover = weights.diff().abs().sum(axis=1)
    turnover.iloc[0] = weights.iloc[0].abs().sum()  # first trade starts from cash

    charged_days = []
    for date, t in turnover.items():
        pos = ret_free.index.get_loc(date)
        if pos + 1 < len(ret_free):
            day = ret_free.index[pos + 1]
            charged_days.append(day)
            assert cost.loc[day] == pytest.approx(t * 0.01)

    # first trade: 70% long + 30% short = 100% turnover, so 1% cost at 100 bps
    assert cost.loc[charged_days[0]] == pytest.approx(0.01)
    assert cost.drop(charged_days).abs().max() < 1e-12


def test_performance_metrics_on_known_series():
    index = pd.bdate_range("2020-01-01", periods=4)
    r = pd.Series([0.0, 0.10, -0.50, 0.20], index=index)
    m = performance_metrics(r)

    assert m["Total Return"] == pytest.approx(1.10 * 0.50 * 1.20 - 1.0)
    assert m["Max Drawdown"] == pytest.approx(-0.50)


def test_sharpe_is_undefined_without_volatility():
    r = pd.Series(0.001, index=pd.bdate_range("2020-01-01", periods=252))
    m = performance_metrics(r)

    assert m["Max Drawdown"] == pytest.approx(0.0)
    assert np.isnan(m["Sharpe (ann.)"])
