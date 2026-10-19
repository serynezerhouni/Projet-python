import numpy as np
import pandas as pd
import pytest

from src.portfolio import compute_weights_from_scores

NAMES = [f"S{i:02d}" for i in range(20)]


@pytest.fixture
def scores():
    return pd.Series(np.linspace(-1.0, 1.0, len(NAMES)), index=NAMES)


@pytest.fixture
def returns():
    rng = np.random.default_rng(7)
    return pd.DataFrame(rng.normal(0.0, 0.02, size=(60, len(NAMES))), columns=NAMES)


@pytest.mark.parametrize("scheme", ["equal", "rank", "inv_vol", "rank_inv_vol", "zscore_inv_vol"])
def test_each_book_hits_its_target_exposure(scheme, scores, returns):
    w = compute_weights_from_scores(scores, returns=returns, weight_scheme=scheme,
                                    long_exposure=0.7, short_exposure=0.3)

    assert w[w > 0].sum() == pytest.approx(0.7)
    assert w[w < 0].sum() == pytest.approx(-0.3)


def test_long_top_20pct_short_bottom_40pct(scores, returns):
    w = compute_weights_from_scores(scores, returns=returns, top_pct=0.2, bottom_pct=0.4)

    assert set(w[w > 0].index) == set(scores.nlargest(4).index)
    assert set(w[w < 0].index) == set(scores.nsmallest(8).index)


def test_unknown_weight_scheme_raises(scores):
    with pytest.raises(ValueError):
        compute_weights_from_scores(scores, weight_scheme="not_a_scheme")
