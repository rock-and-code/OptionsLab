# src/strategies/wizard_signal.py
"""
Price-action signal ported from wizard5919/options_analyzer's
generate_enhanced_signal (lines 1461-1735).

The original strategy combined six weighted conditions: delta, gamma, theta
(options Greeks), plus trend, momentum, volume, and VWAP (price action).
This port keeps the price-action subset only — trend, RSI momentum, and
VWAP — because the Greek conditions are degenerate when evaluated against
a single ATM strike with a fixed sigma (which is what our directional
backtester currently uses). To meaningfully test the Greek layer, we'd
need historical option chains; that's a separate piece of work.

Original weights (price-action portion only):
    trend     0.20
    momentum  0.10
    vwap      0.15
    --------------
    max       0.45

Threshold tuning: wizard's full-strategy threshold was 0.70 / 1.00 (70%).
Applying the same proportion to price-action only: 0.70 * 0.45 = 0.315.
Default `score_threshold=0.315` keeps the comparison faithful.
"""

from typing import Callable, Literal, Optional

import numpy as np
import pandas as pd

CALL_WEIGHTS = {"trend": 0.20, "momentum": 0.10, "vwap": 0.15}
DEFAULT_SCORE_THRESHOLD = 0.315  # 70% of max attainable (0.45)
RSI_WINDOW = 14
EMA_FAST = 9
EMA_SLOW = 20

SignalFn = Callable[[pd.DataFrame], Literal["buy_call", "buy_put", "hold"]]


def compute_indicators(history: pd.DataFrame) -> pd.DataFrame:
    """
    Add EMA_9, EMA_20, RSI, and VWAP columns to a price history DataFrame.

    Required columns: 'Close'. Optional: 'High', 'Low', 'Volume' (used for VWAP
    via typical price * volume). If 'Volume' is missing, VWAP falls back to a
    rolling mean of Close and the VWAP condition becomes a price-vs-trend check.
    """
    df = history.copy()
    df["EMA_9"] = df["Close"].ewm(span=EMA_FAST, adjust=False).mean()
    df["EMA_20"] = df["Close"].ewm(span=EMA_SLOW, adjust=False).mean()

    # Wilder-style RSI (matches the `ta` library wizard used). In a monotone
    # series loss=0; we let division-by-zero produce inf so RSI -> 100, and
    # only treat genuine 0/0 (flat tape) as the neutral RSI=50 case.
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / RSI_WINDOW, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / RSI_WINDOW, adjust=False).mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = gain / loss
        rsi = 100 - 100 / (1 + rs)
    df["RSI"] = rsi.fillna(50.0)

    if {"High", "Low", "Volume"}.issubset(df.columns):
        typical = (df["High"] + df["Low"] + df["Close"]) / 3
        df["VWAP"] = (typical * df["Volume"]).cumsum() / df["Volume"].cumsum()
    else:
        # Fallback: 20-bar mean of Close — preserves the price-vs-mean spirit
        df["VWAP"] = df["Close"].rolling(EMA_SLOW, min_periods=1).mean()

    return df


def score_price_action(
    history_with_indicators: pd.DataFrame, side: Literal["call", "put"]
) -> dict:
    """
    Apply wizard's price-action scoring to the latest bar.

    Returns dict with per-condition pass/fail and total weighted score (0-0.45).
    Caller decides how to threshold the score into a buy/hold decision.
    """
    if history_with_indicators.empty:
        return {"score": 0.0, "trend": False, "momentum": False, "vwap": False}

    last = history_with_indicators.iloc[-1]
    close = float(last["Close"])
    ema_9 = last.get("EMA_9", np.nan)
    ema_20 = last.get("EMA_20", np.nan)
    rsi = last.get("RSI", np.nan)
    vwap = last.get("VWAP", np.nan)

    if side == "call":
        trend_pass = (
            not pd.isna(ema_9) and not pd.isna(ema_20) and close > ema_9 > ema_20
        )
        momentum_pass = not pd.isna(rsi) and rsi > 50
        vwap_pass = not pd.isna(vwap) and close > vwap
    else:
        trend_pass = (
            not pd.isna(ema_9) and not pd.isna(ema_20) and close < ema_9 < ema_20
        )
        momentum_pass = not pd.isna(rsi) and rsi < 50
        vwap_pass = not pd.isna(vwap) and close < vwap

    score = (
        CALL_WEIGHTS["trend"] * trend_pass
        + CALL_WEIGHTS["momentum"] * momentum_pass
        + CALL_WEIGHTS["vwap"] * vwap_pass
    )

    return {
        "score": float(score),
        "trend": bool(trend_pass),
        "momentum": bool(momentum_pass),
        "vwap": bool(vwap_pass),
    }


def make_wizard_signal_fn(
    side: Literal["call", "put"],
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    precomputed_indicators: bool = False,
    max_stretch_pct: Optional[float] = None,
) -> SignalFn:
    """
    Build a signal function ready for DirectionalBacktester.run(...).

    Args:
        side: which direction to trade — fires "buy_call" or "buy_put" when
            the price-action score exceeds the threshold; otherwise "hold".
        score_threshold: minimum score (out of 0.45) required to fire.
        precomputed_indicators: if True, the caller has already added EMA_9/
            EMA_20/RSI/VWAP columns. Otherwise, they are computed on each call.
            Pre-computing once outside the loop is cheaper for long backtests
            but the signal_fn must then receive a slice of the precomputed df.
        max_stretch_pct: if set, suppress the signal when price is more than
            this fraction past VWAP in the signal's direction. E.g. 0.08 means
            don't fire a put when close < VWAP * 0.92, or a call when close >
            VWAP * 1.08. Filters out already-overextended entries that tend
            to mean-revert. None = no filter (faithful to the original wizard
            signal).

    Note:
        compute_indicators is O(n) per call when run inside the bar loop because
        it processes the entire history slice each bar. For multi-year backtests
        prefer precomputed_indicators=True and pre-process the full history.
    """
    target = "buy_call" if side == "call" else "buy_put"

    def signal_fn(history: pd.DataFrame) -> Literal["buy_call", "buy_put", "hold"]:
        if len(history) < EMA_SLOW:
            return "hold"  # not enough bars to trust indicators
        df = history if precomputed_indicators else compute_indicators(history)
        result = score_price_action(df, side)
        if result["score"] < score_threshold:
            return "hold"

        if max_stretch_pct is not None:
            last = df.iloc[-1]
            vwap = last.get("VWAP", np.nan)
            close = float(last["Close"])
            if not pd.isna(vwap) and vwap > 0:
                stretch = (close - vwap) / vwap  # signed: positive=above, negative=below
                if side == "put" and stretch < -max_stretch_pct:
                    return "hold"
                if side == "call" and stretch > max_stretch_pct:
                    return "hold"

        return target

    return signal_fn
