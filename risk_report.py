# risk_report.py
"""
Market risk report on the strategy's daily returns.

Runs offline on the returns saved by main.py (no data download needed):

    python risk_report.py                                   # uses outs/tables/returns_final.csv
    python risk_report.py --returns outs/tables/returns_oos_2020_2024.csv --tag oos
    python risk_report.py --market path/to/market_returns.csv   # adds rolling beta

Outputs
- outs/tables/risk_var_summary[_tag].csv      VaR / ES by method (95%, 99%, 1d and 10d)
- outs/tables/risk_stressed_var[_tag].csv     Stressed VaR on the worst 12-month window
- outs/tables/risk_var_backtest[_tag].csv     Kupiec / Christoffersen / Basel traffic light
- outs/tables/risk_stress_scenarios[_tag].csv Historical stress windows replayed on the P&L
- outs/figures/risk_*.png                     Distribution, VaR backtest, rolling volatility
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.risk import (
    clean_returns,
    historical_es,
    historical_var,
    parametric_var,
    rolling_beta,
    rolling_volatility,
    scale_to_horizon,
    stress_scenarios,
    stressed_var,
    var_backtest,
    var_summary,
)

OUT_DIR = Path("outs")
FIG_DIR = OUT_DIR / "figures"
TAB_DIR = OUT_DIR / "tables"


def load_returns(path: Path) -> pd.Series:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df.iloc[:, 0].astype(float).rename("strategy")


def plot_distribution(r: pd.Series, path: Path) -> None:
    var99 = historical_var(r, 0.99)
    es99 = historical_es(r, 0.99)
    pvar99 = parametric_var(r, 0.99)

    plt.figure(figsize=(8, 4.5))
    plt.hist(r * 100, bins=120, color="#9db4d0", edgecolor="none")
    plt.axvline(-var99 * 100, color="#1f4e79", lw=1.6, label=f"Historical VaR 99%: {var99:.2%}")
    plt.axvline(-pvar99 * 100, color="#1f4e79", lw=1.2, ls="--", label=f"Gaussian VaR 99%: {pvar99:.2%}")
    plt.axvline(-es99 * 100, color="#b22222", lw=1.6, label=f"Expected Shortfall 99%: {es99:.2%}")
    plt.title("Daily return distribution with 1-day VaR and ES")
    plt.xlabel("Daily return (%)")
    plt.ylabel("Number of days")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    print(f" saved {path}")


def plot_var_backtest(bt: pd.DataFrame, alpha: float, path: Path) -> None:
    exc = bt[bt["exception"]]
    plt.figure(figsize=(10, 4.5))
    plt.plot(bt.index, bt["return"] * 100, color="#9db4d0", lw=0.6, label="Daily return")
    plt.plot(bt.index, -bt["VaR"] * 100, color="#1f4e79", lw=1.2, label=f"-VaR {int(alpha * 100)}% (rolling 250d)")
    plt.scatter(exc.index, exc["return"] * 100, color="#b22222", s=10, zorder=3,
                label=f"Exceptions ({len(exc)})")
    plt.title(f"VaR backtest: historical {int(alpha * 100)}% one-day VaR vs realised returns")
    plt.ylabel("%")
    plt.legend(frameon=False, loc="lower left")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    print(f" saved {path}")


def plot_rolling_vol(r: pd.Series, path: Path) -> None:
    vol = rolling_volatility(r, 63)
    plt.figure(figsize=(10, 3.8))
    plt.plot(vol.index, vol * 100, color="#1f4e79", lw=1.0)
    plt.title("Rolling 3-month volatility (annualised)")
    plt.ylabel("%")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    print(f" saved {path}")


def run_report(returns_path, market_path=None, tag: str = "") -> None:
    """Builds every risk table and figure from a CSV of daily strategy returns."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    sfx = f"_{tag}" if tag else ""

    r = clean_returns(load_returns(Path(returns_path)))
    print(f" {len(r)} daily returns from {r.index[0].date()} to {r.index[-1].date()}")

    # 1) VaR / ES
    summary = var_summary(r, alphas=(0.95, 0.99), horizon_days=10)
    summary.to_csv(TAB_DIR / f"risk_var_summary{sfx}.csv")
    print(summary.round(4))

    # 2) Stressed VaR (worst 12-month window)
    svar = stressed_var(r, alpha=0.99)
    svar["stressed_var_10d"] = scale_to_horizon(svar["stressed_var"], 10)
    pd.DataFrame([svar]).to_csv(TAB_DIR / f"risk_stressed_var{sfx}.csv", index=False)
    print(svar)

    # 3) VaR backtesting
    rows, bt99 = [], None
    for a in (0.95, 0.99):
        s, bt = var_backtest(r, alpha=a, window=250, method="historical")
        rows.append(s)
        if a == 0.99:
            bt99 = bt
    bt_table = pd.DataFrame(rows)
    bt_table.to_csv(TAB_DIR / f"risk_var_backtest{sfx}.csv", index=False)
    print(bt_table.round(4).to_string(index=False))

    # 4) Historical stress scenarios
    st = stress_scenarios(r)
    st.to_csv(TAB_DIR / f"risk_stress_scenarios{sfx}.csv")
    print(st.round(4))

    # 5) Figures
    plot_distribution(r, FIG_DIR / f"risk_return_distribution{sfx}.png")
    plot_var_backtest(bt99, 0.99, FIG_DIR / f"risk_var_backtest{sfx}.png")
    plot_rolling_vol(r, FIG_DIR / f"risk_rolling_vol{sfx}.png")

    # 6) Optional rolling beta
    if market_path:
        m = pd.read_csv(market_path, index_col=0, parse_dates=True).iloc[:, 0]
        beta = rolling_beta(r, m, window=126)
        beta.to_csv(TAB_DIR / f"risk_rolling_beta{sfx}.csv", header=True)
        plt.figure(figsize=(10, 3.8))
        plt.plot(beta.index, beta, color="#1f4e79", lw=1.0)
        plt.axhline(0, color="grey", lw=0.6)
        plt.title("Rolling 6-month beta to the S&P 500")
        plt.tight_layout()
        plt.savefig(FIG_DIR / f"risk_rolling_beta{sfx}.png", dpi=200)
        plt.close()
        print(f" saved {FIG_DIR / f'risk_rolling_beta{sfx}.png'}")

    print("\n Risk report done")


def main() -> None:
    parser = argparse.ArgumentParser(description="Market risk report on strategy returns")
    parser.add_argument("--returns", default=str(TAB_DIR / "returns_final.csv"))
    parser.add_argument("--market", default=None, help="Optional CSV of daily market returns (date, return)")
    parser.add_argument("--tag", default="", help="Suffix added to output file names")
    args = parser.parse_args()
    run_report(args.returns, market_path=args.market, tag=args.tag)


if __name__ == "__main__":
    main()
