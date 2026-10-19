import numpy as np
import pandas as pd

from src.signals import momentum_scores


def test_scores_only_use_data_up_to_the_scoring_date(market):
    prices, volumes = market
    as_of = prices.index[400]

    full = momentum_scores(prices, volumes, lookback_days=60, as_of_date=as_of)
    truncated = momentum_scores(prices.loc[:as_of], volumes.loc[:as_of], lookback_days=60, as_of_date=as_of)

    pd.testing.assert_series_equal(full, truncated)


def test_future_price_shock_does_not_change_scores(market):
    prices, volumes = market
    as_of = prices.index[400]

    shocked = prices.copy()
    shocked.iloc[401:] = shocked.iloc[401:] * 3.0

    before = momentum_scores(prices, volumes, lookback_days=60, as_of_date=as_of)
    after = momentum_scores(shocked, volumes, lookback_days=60, as_of_date=as_of)

    pd.testing.assert_series_equal(before, after)


def test_stronger_trend_gets_higher_score():
    # stock k grows by k basis points a day: the ranking must follow k
    index = pd.bdate_range("2018-01-01", periods=300)
    growth = np.arange(1, 11) * 1e-4
    columns = [f"S{k}" for k in range(1, 11)]
    prices = pd.DataFrame(100.0 * np.exp(np.outer(np.arange(300), growth)), index=index, columns=columns)
    volumes = pd.DataFrame(1e6, index=index, columns=columns)

    scores = momentum_scores(
        prices, volumes, lookback_days=60,
        risk_adjust_by_vol=False, use_rsi=False, use_volume_penalty=False,
    )

    assert list(scores.sort_values().index) == columns


def test_illiquid_stock_is_penalised(market):
    prices, volumes = market
    # two identical price paths, one traded below the volume threshold
    prices = prices[["S00"]].assign(S00_illiquid=prices["S00"], S01=prices["S01"])
    volumes = pd.DataFrame(1e6, index=prices.index, columns=prices.columns)
    volumes["S00_illiquid"] = 1e5

    scores = momentum_scores(prices, volumes, lookback_days=60, use_rsi=False, volume_penalty_factor=0.5)

    assert scores["S00_illiquid"] == np.float64(0.5) * scores["S00"]
