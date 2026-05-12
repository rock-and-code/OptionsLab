# src/backtesting/directional_engine.py
"""
Directional backtesting engine for options strategies.

Replays historical prices and evaluates a user-supplied signal function
on every bar. When the signal fires, opens an ATM call or put, marks
to market each subsequent bar via Black-Scholes, and exits on profit
target, stop loss, max hold period, or expiry.

Usage:
    >>> from src.backtesting import DirectionalBacktester
    >>> def buy_and_hold(history): return "buy_call" if len(history) == 1 else "hold"
    >>> bt = DirectionalBacktester()
    >>> result = bt.run("SPY", "2024-01-01", "2024-06-01", buy_and_hold)
"""

from dataclasses import dataclass
from typing import Callable, Literal, Optional

import numpy as np
import pandas as pd

from src.backtesting.backtest_engine import BacktestResult
from src.pricing_models.black_scholes import black_scholes

try:
    import yfinance as yf

    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False


SignalFn = Callable[[pd.DataFrame], Literal["buy_call", "buy_put", "hold"]]
TRADING_DAYS_PER_YEAR = 252


def rolling_realized_vol(
    closes: pd.Series, window: int = 20, annualize: bool = True
) -> pd.Series:
    """
    Annualized rolling realized volatility from log returns.

    The first `window` bars are NaN. Callers should fall back to a constant
    when looking up sigma on early bars.
    """
    log_ret = np.log(closes / closes.shift(1))
    vol = log_ret.rolling(window).std()
    if annualize:
        vol = vol * np.sqrt(TRADING_DAYS_PER_YEAR)
    return vol


@dataclass
class _OpenPosition:
    side: Literal["call", "put"]
    strike: float
    entry_price: float
    entry_idx: int
    expiry_idx: int
    sigma_at_entry: float  # locked at open so MTM is consistent with entry


class DirectionalBacktester:
    """Replays history bar-by-bar, executes a signal function, books P&L."""

    def __init__(
        self,
        risk_free_rate: float = 0.05,
        sigma: float = 0.20,
        profit_target_pct: float = 0.15,
        stop_loss_pct: float = 0.08,
        dte_days: int = 30,
        max_hold_days: int = 10,
    ):
        self.r = risk_free_rate
        self.sigma = sigma  # constant fallback when no sigma_series is provided
        self.profit_target_pct = profit_target_pct
        self.stop_loss_pct = stop_loss_pct
        self.dte_days = dte_days
        self.max_hold_days = max_hold_days

    @staticmethod
    def _fetch_history(ticker: str, start: str, end: str) -> pd.DataFrame:
        if not YFINANCE_AVAILABLE:
            raise ImportError("yfinance required: pip install yfinance")
        hist = yf.Ticker(ticker).history(start=start, end=end)
        if hist.empty:
            raise ValueError(f"No data for {ticker} from {start} to {end}")
        return hist

    def _option_price(
        self, side: str, S: float, K: float, days_to_expiry: int, sigma: float
    ) -> float:
        T = max(days_to_expiry / TRADING_DAYS_PER_YEAR, 1e-6)
        return black_scholes(S=S, K=K, T=T, r=self.r, sigma=sigma, option_type=side)

    def _open_position(
        self, signal: str, idx: int, spot: float, n_bars: int, sigma: float
    ) -> _OpenPosition:
        side = "call" if signal == "buy_call" else "put"
        strike = spot  # ATM
        # Price entry against the option's *actual* life so subsequent mark-to-
        # market (which uses expiry_idx - idx) is consistent with the entry.
        # Without this, positions opened near the data end appear to instantly
        # lose theta because entry was priced at the full dte_days.
        expiry_idx = min(idx + self.dte_days, n_bars - 1)
        days_to_expiry = expiry_idx - idx
        entry_price = self._option_price(side, spot, strike, days_to_expiry, sigma)
        return _OpenPosition(
            side=side,
            strike=strike,
            entry_price=entry_price,
            entry_idx=idx,
            expiry_idx=expiry_idx,
            sigma_at_entry=sigma,
        )

    def _check_exit(
        self, pos: _OpenPosition, idx: int, spot: float
    ) -> Optional[tuple[float, str]]:
        """Return (exit_price, reason) if the position should close on this bar."""
        days_held = idx - pos.entry_idx
        days_to_expiry = pos.expiry_idx - idx

        if days_to_expiry <= 0:
            payoff = max(spot - pos.strike, 0) if pos.side == "call" else max(
                pos.strike - spot, 0
            )
            return payoff, "expiry"

        current_price = self._option_price(
            pos.side, spot, pos.strike, days_to_expiry, pos.sigma_at_entry
        )
        pnl_pct = (current_price - pos.entry_price) / pos.entry_price

        if pnl_pct <= -self.stop_loss_pct:
            return current_price, "stop_loss"
        if pnl_pct >= self.profit_target_pct:
            return current_price, "profit_target"
        if days_held >= self.max_hold_days:
            return current_price, "max_hold"
        return None

    def run(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        signal_fn: SignalFn,
        price_history: Optional[pd.DataFrame] = None,
        sigma_series: Optional[pd.Series] = None,
    ) -> BacktestResult:
        """
        Run a directional backtest.

        Args:
            ticker, start_date, end_date: forwarded to yfinance unless price_history given.
            signal_fn: called with the price history slice up to (and including) each bar.
                Must return "buy_call", "buy_put", or "hold". A new signal is ignored
                while a position is open.
            price_history: optional pre-fetched DataFrame with a 'Close' column. If
                provided, ticker/start/end are used only for labeling.
            sigma_series: optional per-bar volatility (annualized). Must have the same
                length as price_history. Used at position entry; once a trade is open,
                sigma is locked. NaNs and non-positive values fall back to self.sigma
                (the default fallback covers the warm-up period of rolling estimates).

        Returns:
            BacktestResult populated with per-bar P&L and summary stats.
        """
        hist = price_history if price_history is not None else self._fetch_history(
            ticker, start_date, end_date
        )
        prices = hist["Close"].values
        dates = (
            hist.index.strftime("%Y-%m-%d").tolist()
            if hasattr(hist.index, "strftime")
            else [str(i) for i in range(len(hist))]
        )
        n_bars = len(prices)
        if n_bars < 2:
            raise ValueError("Need at least 2 bars of price history")

        if sigma_series is not None and len(sigma_series) != n_bars:
            raise ValueError(
                f"sigma_series length {len(sigma_series)} != price history length {n_bars}"
            )

        def sigma_at(i: int) -> float:
            if sigma_series is None:
                return self.sigma
            v = float(sigma_series.iloc[i])
            return v if (v > 0 and not np.isnan(v)) else self.sigma

        position: Optional[_OpenPosition] = None
        trades: list[float] = []  # realized P&L per closed trade
        daily_pnl: list[float] = [0.0] * n_bars  # mark-to-market change per bar
        last_mark: float = 0.0

        for i in range(n_bars):
            spot = prices[i]

            if position is not None:
                exit_check = self._check_exit(position, i, spot)
                if exit_check is not None:
                    exit_price, _reason = exit_check
                    new_mark = exit_price - position.entry_price
                    daily_pnl[i] = new_mark - last_mark
                    trades.append(new_mark)
                    position = None
                    last_mark = 0.0
                else:
                    days_to_expiry = position.expiry_idx - i
                    current_price = self._option_price(
                        position.side,
                        spot,
                        position.strike,
                        days_to_expiry,
                        position.sigma_at_entry,
                    )
                    new_mark = current_price - position.entry_price
                    daily_pnl[i] = new_mark - last_mark
                    last_mark = new_mark
                continue

            # Avoid end-of-data lookahead bias: don't open a position that
            # cannot reach max_hold_days within the remaining series. Without
            # this guard, trades opened near the end exit early at intrinsic
            # value (truncated expiry) and bias win rate downward.
            bars_remaining = n_bars - 1 - i
            if bars_remaining >= self.max_hold_days:
                signal = signal_fn(hist.iloc[: i + 1])
                if signal in ("buy_call", "buy_put"):
                    position = self._open_position(
                        signal, i, spot, n_bars, sigma_at(i)
                    )
                    last_mark = 0.0

        cumulative_pnl = np.cumsum(daily_pnl).tolist()
        total_pnl = float(sum(trades))
        n_trades = len(trades)
        win_rate = float(np.mean([t > 0 for t in trades])) if trades else 0.0

        daily_arr = np.array(daily_pnl)
        if daily_arr.std() > 0:
            sharpe = float(daily_arr.mean() / daily_arr.std() * np.sqrt(TRADING_DAYS_PER_YEAR))
        else:
            sharpe = 0.0

        cum = np.array(cumulative_pnl)
        if len(cum) > 0:
            peaks = np.maximum.accumulate(cum)
            drawdowns = peaks - cum
            max_dd = float(drawdowns.max())
        else:
            max_dd = 0.0

        initial_value = 1000.0  # nominal — directional bt has no premium-received concept
        return BacktestResult(
            strategy="Directional (signal-driven)",
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            initial_value=initial_value,
            final_value=initial_value + total_pnl,
            total_pnl=total_pnl,
            total_return_pct=(total_pnl / initial_value * 100) if initial_value else 0.0,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            win_rate=win_rate,
            n_trades=n_trades,
            daily_pnl=daily_pnl,
            cumulative_pnl=cumulative_pnl,
            dates=dates,
            model_greeks={"sigma": self.sigma, "dte_days": self.dte_days},
        )
