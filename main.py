# main.py
from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from src.data import get_prices_and_volume
from src.backtest import backtest_pocheA_momentum
from src.analysis import (
    compare_exposures_capm,
    analyse_ff3_sur_strategie,
    run_subperiods_table,
    hyperparam_grid_search,
)

# =========================
# 0) Dossiers de sortie
# =========================
OUT_DIR = Path("outs")
FIG_DIR = OUT_DIR / "figures"
TAB_DIR = OUT_DIR / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

def save_df(df: pd.DataFrame, name: str):
    path = TAB_DIR / name
    df.to_csv(path, index=True)

def equity_curve(returns: pd.Series) -> pd.Series:
    returns = returns.fillna(0.0)
    return (1.0 + returns).cumprod()

def drawdown_curve(equity: pd.Series) -> pd.Series:
    peak = equity.cummax()
    return equity / peak - 1.0

def plot_equity(returns: pd.Series, filename: str, title: str):
    eq = equity_curve(returns)
    plt.figure()
    plt.plot(eq.index, eq.values)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(FIG_DIR / filename, dpi=200)
    plt.close()

def plot_drawdown(returns: pd.Series, filename: str, title: str):
    eq = equity_curve(returns)
    dd = drawdown_curve(eq)
    plt.figure()
    plt.plot(dd.index, dd.values)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(FIG_DIR / filename, dpi=200)
    plt.close()


# =========================
# 1) PARAMS FINAL (produit final)
# =========================
FINAL_PARAMS = dict(
    lookback_days=60,
    top_pct=0.2,
    bottom_pct=0.4,
    long_exposure=0.7,
    short_exposure=0.3,
    rebal_freq="M",
    transaction_cost_bps=8.0,
    vol_window=20,
    max_weight=None,
    min_names_per_side=3,
    weight_scheme="rank_inv_vol",
    risk_adjust_by_vol_in_score=False,
)

# Univers & dates : si ton src/data.py contient déjà l’univers + dates, garde-les là-bas.
# Sinon, ajuste ici selon ton data.py.
START_DATE = "2010-01-01"
END_DATE = None


def main():
    # 1) Data
    prices, volumes = get_prices_and_volume(start=START_DATE, end=END_DATE)
    print(f"Data: {prices.shape[0]} dates x {prices.shape[1]} tickers")

    # 2) Backtest FINAL
    ret_final, w_final, met_final = backtest_pocheA_momentum(prices, volumes, **FINAL_PARAMS)

    save_df(pd.DataFrame([met_final], index=["final"]), "perf_final.csv")
    ret_final.to_csv(TAB_DIR / "returns_final.csv", header=True)
    plot_equity(ret_final, "equity_final.png", "Equity curve — Final")
    plot_drawdown(ret_final, "drawdown_final.png", "Drawdown — Final")

    # 3) Comparaison Equal vs Final (rank/inv-vol)
    params_equal = FINAL_PARAMS.copy()
    params_equal["weight_scheme"] = "equal"
    ret_eq, w_eq, met_eq = backtest_pocheA_momentum(prices, volumes, **params_equal)

    cmp = pd.concat([
        pd.DataFrame([met_eq], index=["equal_weight"]),
        pd.DataFrame([met_final], index=["rank_inv_vol"]),
    ])
    save_df(cmp, "weighting_comparison.csv")

    # figures comparaison
    plt.figure()
    plt.plot(equity_curve(ret_eq), label="equal_weight")
    plt.plot(equity_curve(ret_final), label="rank_inv_vol")
    plt.title("Equity curve — Equal vs Rank/Inv-Vol")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "equity_equal_vs_invvol.png", dpi=200)
    plt.close()

    # 4) Sous-périodes
    sub = run_subperiods_table(
        prices, volumes,
        periods=[("2010-01-01","2014-12-31"), ("2015-01-01","2019-12-31"), ("2020-01-01","2024-12-31")],
        long_expo=0.7, short_expo=0.3,
        top_pct=0.2, bottom_pct=0.4,
        vol_window=20, rebal_freq="M",
        transaction_cost_bps=8.0,
        max_weight=None, min_names_per_side=3,
        weight_scheme="rank_inv_vol",
        risk_adjust_by_vol_in_score=False
    )
    save_df(sub, "subperiods_table.csv")

    # 5) Sensibilité hyperparams
    grid = hyperparam_grid_search(
        prices, volumes,
        top_list=(0.1, 0.2),
        bottom_list=(0.2, 0.4),
        exposure_list=((0.5,0.5), (0.7,0.3)),
        vol_window=20, rebal_freq="M",
        transaction_cost_bps=8.0,
        max_weight=None, min_names_per_side=3,
        weight_scheme="rank_inv_vol",
        risk_adjust_by_vol_in_score=False
    )
    save_df(grid, "sensitivity_grid_full.csv")
    save_df(grid.sort_values("Sharpe", ascending=False).head(10), "sensitivity_best_top10.csv")

    # 6) CAPM 70/30 vs 50/50
    capm_cmp = compare_exposures_capm(
        prices, volumes,
        top_pct=0.2, bottom_pct=0.4,
        vol_window=20, rebal_freq="M",
        transaction_cost_bps=8.0,
        max_weight=None, min_names_per_side=3,
        weight_scheme="rank_inv_vol",
        risk_adjust_by_vol_in_score=False
    )
    save_df(capm_cmp, "capm_7030_vs_5050.csv")

    # 7) FF3 (sur final)
    ret_ff, met_ff, ff_stats = analyse_ff3_sur_strategie(
        prices, volumes,
        long_expo=0.7, short_expo=0.3,
        top_pct=0.2, bottom_pct=0.4,
        vol_window=20, rebal_freq="M",
        transaction_cost_bps=8.0,
        max_weight=None, min_names_per_side=3,
        weight_scheme="rank_inv_vol",
        risk_adjust_by_vol_in_score=False
    )
    save_df(pd.DataFrame([ff_stats], index=["ff3_final"]), "ff3_loadings_final.csv")

    print("✅ Terminé. Résultats dans outs/figures et outs/tables")


if __name__ == "__main__":
    main()
