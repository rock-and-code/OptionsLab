# src/experiments/wizard_signal_walkforward.py
"""
Walk-forward evaluation of the wizard5919-derived directional strategy
across a panel of (ticker, year) pairs with FIXED parameters.

This is the honest measurement: no per-pair tuning, no parameter sweeps
inside the loop, just one set of params applied uniformly. The output
tells us whether the strategy generalizes across regimes (bull / bear /
chop) and underlyings (broad market / tech / single names / commodities).

Usage:
    python -m src.experiments.wizard_signal_walkforward
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from src.backtesting import DirectionalBacktester, rolling_realized_vol
from src.strategies.wizard_signal import make_wizard_signal_fn


@dataclass
class PanelRow:
    ticker: str
    year: int
    side: str  # "call" or "put"
    underlying_return_pct: float
    n_trades: int
    win_rate: float
    total_pnl: float
    sharpe: float
    max_dd: float


def fetch_panel(
    tickers: list[str], years: list[int]
) -> dict[tuple[str, int], pd.DataFrame]:
    """Fetch one year of history per (ticker, year). Skips empty results."""
    import yfinance as yf  # imported here so tests can avoid the network dep

    out: dict[tuple[str, int], pd.DataFrame] = {}
    for t in tickers:
        for y in years:
            h = yf.Ticker(t).history(start=f"{y}-01-01", end=f"{y + 1}-01-01")
            if not h.empty:
                out[(t, y)] = h
    return out


def run_walkforward(
    panel_data: dict[tuple[str, int], pd.DataFrame],
    *,
    sides: tuple[str, ...] = ("call", "put"),
    max_stretch_pct: Optional[float] = 0.08,
    profit_target_pct: float = 0.15,
    stop_loss_pct: float = 0.08,
    max_hold_days: int = 10,
    dte_days: int = 30,
    sigma_window: int = 20,
) -> pd.DataFrame:
    """
    Evaluate the wizard signal on every (ticker, year) pair in panel_data.

    Returns a DataFrame with one row per (ticker, year, side).
    """
    rows: list[PanelRow] = []
    for (ticker, year), hist in panel_data.items():
        if len(hist) < sigma_window + max_hold_days:
            continue
        sigma_series = rolling_realized_vol(hist["Close"], window=sigma_window)
        underlying_ret = float(hist["Close"].iloc[-1] / hist["Close"].iloc[0] - 1)

        for side in sides:
            signal_fn = make_wizard_signal_fn(side, max_stretch_pct=max_stretch_pct)
            bt = DirectionalBacktester(
                profit_target_pct=profit_target_pct,
                stop_loss_pct=stop_loss_pct,
                max_hold_days=max_hold_days,
                dte_days=dte_days,
            )
            r = bt.run(
                ticker,
                f"{year}-01-01",
                f"{year + 1}-01-01",
                signal_fn,
                price_history=hist,
                sigma_series=sigma_series,
            )
            rows.append(
                PanelRow(
                    ticker=ticker,
                    year=year,
                    side=side,
                    underlying_return_pct=underlying_ret * 100,
                    n_trades=r.n_trades,
                    win_rate=r.win_rate,
                    total_pnl=r.total_pnl,
                    sharpe=r.sharpe_ratio,
                    max_dd=r.max_drawdown,
                )
            )

    return pd.DataFrame([row.__dict__ for row in rows])


def summarize(panel_df: pd.DataFrame) -> dict:
    """Aggregate stats. Trade-weighted (not equal-weighted across pairs)."""
    out: dict = {}
    for side in panel_df["side"].unique():
        slc = panel_df[panel_df["side"] == side]
        total_trades = int(slc["n_trades"].sum())
        # Weighted win rate: total wins across all pairs / total trades.
        # (n_trades * win_rate is wins for that pair.)
        wins = int((slc["n_trades"] * slc["win_rate"]).sum())
        out[side] = {
            "pairs": int(len(slc)),
            "total_trades": total_trades,
            "weighted_win_rate": wins / total_trades if total_trades else 0.0,
            "summed_pnl": float(slc["total_pnl"].sum()),
            "mean_sharpe": float(slc["sharpe"].mean()),
            "positive_pair_pct": float((slc["total_pnl"] > 0).mean()),
        }
    return out


def _print_panel(df: pd.DataFrame) -> None:
    """Compact per-row view."""
    print(
        f"{'ticker':<6} {'year':>4} {'side':>4} {'undr%':>7} "
        f"{'trades':>6} {'win%':>6} {'pnl':>8} {'sharpe':>7} {'max_dd':>7}"
    )
    for _, r in df.iterrows():
        print(
            f"{r.ticker:<6} {r.year:>4} {r.side:>4} "
            f"{r.underlying_return_pct:>+7.1f} {r.n_trades:>6d} "
            f"{r.win_rate * 100:>5.1f}% {r.total_pnl:>+8.2f} "
            f"{r.sharpe:>+7.2f} {r.max_dd:>7.2f}"
        )


if __name__ == "__main__":
    tickers = ["SPY", "QQQ", "IWM", "AAPL", "TSLA", "GLD"]
    years = list(range(2019, 2025))

    print(f"Fetching {len(tickers)} tickers x {len(years)} years...")
    panel = fetch_panel(tickers, years)
    print(f"  got {len(panel)} pairs\n")

    print("Running walk-forward (fixed params: stretch=0.08, target=15%, stop=8%, "
          "max_hold=10, dte=30, sigma=20-day realized)...\n")
    df = run_walkforward(panel)
    _print_panel(df)

    print()
    print("Aggregate:")
    summary = summarize(df)
    for side, stats in summary.items():
        print(
            f"  {side:>4}: "
            f"pairs={stats['pairs']:>2}, trades={stats['total_trades']:>4}, "
            f"win={stats['weighted_win_rate'] * 100:>4.1f}%, "
            f"pnl=${stats['summed_pnl']:>+8.2f}, "
            f"mean_sharpe={stats['mean_sharpe']:>+5.2f}, "
            f"pos_pair_pct={stats['positive_pair_pct'] * 100:>4.1f}%"
        )
