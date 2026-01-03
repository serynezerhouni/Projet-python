# main.py
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.data import get_prices_and_volume
from src.backtest import backtest_pocheA_momentum
from src.analysis import (
    compare_exposures_capm,
    analyse_ff3_sur_strategie,
    hyperparam_grid_search,
    run_subperiods_table,
    run_ablation_tests,
)

# =========================
# 1) CONFIG PROJET
# =========================

# Univers (mets exactement ta liste)
UNIVERSE = [
    "AAPL","ABBV","ABT","ACN","ADBE","AIG","AMD","AMGN","AMT","AMZN","AVGO","AXP","BA","BAC","BK",
    "BKNG","BLK","BMY","BRK-B","C","CAT","CL","CMCSA","COF","COP","COST","CRM","CSCO","CVS","CVX",
    "DE","DHR","DIS","DUK","EMR","FDX","GD","GE","GILD","GM","GOOG","GOOGL","GS","HD","HON","IBM",
    "INTC","INTU","ISRG","JNJ","JPM","KO","LIN","LLY","LMT","LOW","MA","MCD","MDLZ","MDT","MET",
    "META","MMM","MRK","MS","MSFT","NEE","NFLX","NKE","NOW","NVDA","ORCL","PEP","PFE","PG","PLTR",
    "PM","PYPL","QCOM","RTX","SBUX","SCHW","SO","SPG","T","TGT","TMO","TMUS","TSLA","TXN","UBER",
    "UNH","UNP","UPS","USB","V","VZ","WFC","WMT","XOM"
]

START_DATE = "2010-01-01"
END_DATE = None  # None = jusqu'à aujourd'hui

# Produit final (ce que tu m’as donné)
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

# =========================
# 2) DOSSIERS OUTPUT
# =========================
OUT_DIR = Path("outs")
FIG_DIR = OUT_DIR / "figures"
TAB_DIR = OUT_DIR / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)


def save_df(df: pd.DataFrame, name: str):
    path = TAB_DIR / name
    df.to_csv(path, index=True)
    print(f"✅ saved {path}")


def save_series(s: pd.Series, name: str):
    path = TAB_DIR / name
    s.to_csv(path, header=True)
    print(f"✅ saved {path}")


def metrics_dict_to_df(metrics: dict, label: str) -> pd.DataFrame:
    return pd.DataFrame([{"label": label, **metrics}]).set_index("label")


def equity_curve(returns: pd.Series) -> pd.Series:
    return (1.0 + returns.fillna(0.0)).cumprod()


def drawdown_curve(equity: pd.Series) -> pd.Series:
    peak = equity.cummax()
    return equity / peak - 1.0


def plot_and_save_equity(ret_dict: dict, filename: str, title: str):
    plt.figure()
    for k, r in ret_dict.items():
        eq = equity_curve(r)
        plt.plot(eq.index, eq.values, label=k)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / filename, dpi=200)
    plt.close()
    print(f"✅ saved {FIG_DIR/filename}")


def plot_and_save_drawdown(ret_dict: dict, filename: str, title: str):
    plt.figure()
    for k, r in ret_dict.items():
        eq = equity_curve(r)
        dd = drawdown_curve(eq)
        plt.plot(dd.index, dd.values, label=k)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / filename, dpi=200)
    plt.close()
    print(f"✅ saved {FIG_DIR/filename}")


# =========================
# 3) MAIN PIPELINE
# =========================
def main():
    # 1) Data (IMPORTANT: on passe tickers=UNIVERSE)
    prices, volumes = get_prices_and_volume(
        tickers=UNIVERSE,
        start=START_DATE,
        end=END_DATE,
    )
    print(f"✅ Prix téléchargés: {prices.shape[0]} dates x {prices.shape[1]} tickers")

    # 2) Backtest final + exports
    ret_final, w_final, met_final = backtest_pocheA_momentum(
        prices=prices,
        volumes=volumes,
        **FINAL_PARAMS,
    )

    save_series(ret_final, "returns_final.csv")
    save_df(metrics_dict_to_df(met_final, "final"), "perf_final.csv")
    plot_and_save_equity({"final": ret_final}, "equity_final.png", "Equity curve — Final")
    plot_and_save_drawdown({"final": ret_final}, "drawdown_final.png", "Drawdown — Final")

    # 3) Comparaison equal-weight vs final (mêmes params sinon)
    params_equal = FINAL_PARAMS.copy()
    params_equal["weight_scheme"] = "equal"
    # (le score reste non risk-adjust; on compare juste la pondération)
    ret_equal, w_equal, met_equal = backtest_pocheA_momentum(
        prices=prices,
        volumes=volumes,
        **params_equal,
    )

    cmp = pd.concat(
        [
            metrics_dict_to_df(met_equal, "equal_weight"),
            metrics_dict_to_df(met_final, "rank_inv_vol"),
        ],
        axis=0,
    )
    save_df(cmp, "weighting_comparison.csv")

    plot_and_save_equity(
        {"equal_weight": ret_equal, "rank_inv_vol": ret_final},
        "equity_equal_vs_invvol.png",
        "Equity curve — Equal-weight vs Rank/Inv-Vol",
    )
    plot_and_save_drawdown(
        {"equal_weight": ret_equal, "rank_inv_vol": ret_final},
        "drawdown_equal_vs_invvol.png",
        "Drawdown — Equal-weight vs Rank/Inv-Vol",
    )

    # 4) Sensibilité hyperparamètres
    grid = hyperparam_grid_search(
        prices=prices,
        volumes=volumes,
        top_list=(0.1, 0.2),
        bottom_list=(0.2, 0.4),
        exposure_list=((0.5, 0.5), (0.7, 0.3)),
        weight_scheme="rank_inv_vol",
        risk_adjust_by_vol_in_score=False,
    )
    save_df(grid, "sensitivity_grid_full.csv")
    save_df(grid.sort_values("Sharpe", ascending=False).head(10), "sensitivity_best_top10.csv")

    # 5) Sous-périodes
    sub = run_subperiods_table(
        prices=prices,
        volumes=volumes,
        periods=[
            ("2010-01-01", "2014-12-31"),
            ("2015-01-01", "2019-12-31"),
            ("2020-01-01", "2024-12-31"),
        ],
        long_expo=0.7,
        short_expo=0.3,
        weight_scheme="rank_inv_vol",
        risk_adjust_by_vol_in_score=False,
    )
    save_df(sub, "subperiods_table.csv")

    # 6) CAPM: 70/30 vs 50/50
    capm_cmp = compare_exposures_capm(
        prices=prices,
        volumes=volumes,
        weight_scheme="rank_inv_vol",
        risk_adjust_by_vol_in_score=False,
    )
    save_df(capm_cmp, "capm_7030_vs_5050.csv")

    # 7) FF3 sur la stratégie finale
    ret_ff, met_ff, ff_stats = analyse_ff3_sur_strategie(
        prices=prices,
        volumes=volumes,
        long_expo=0.7,
        short_expo=0.3,
        weight_scheme="rank_inv_vol",
        risk_adjust_by_vol_in_score=False,
    )
    ff_df = pd.DataFrame([ff_stats])
    save_df(ff_df, "ff3_loadings_final.csv")

    # 8) Ablation
    ablation = run_ablation_tests(prices, volumes)
    save_df(ablation.set_index("variant"), "ablation_final.csv")

    # Barplot Sharpe ablation
    plt.figure()
    plt.bar(ablation["variant"], ablation["Sharpe"])
    plt.xticks(rotation=30, ha="right")
    plt.title("Ablation — Sharpe par variante")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "ablation_sharpe_bar.png", dpi=200)
    plt.close()
    print(f"✅ saved {FIG_DIR/'ablation_sharpe_bar.png'}")

    print("\n✅ FIN : regarde outs/figures et outs/tables")


if __name__ == "__main__":
    main()
