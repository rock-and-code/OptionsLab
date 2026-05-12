import numpy as np
import pandas as pd
import pytest

from src.backtesting import DirectionalBacktester
from src.strategies.wizard_signal import (
    compute_indicators,
    make_wizard_signal_fn,
    score_price_action,
)


def _trending_up(n: int = 60, slope: float = 0.5) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    close = 100.0 + np.arange(n) * slope
    return pd.DataFrame(
        {"Close": close, "High": close + 0.2, "Low": close - 0.2, "Volume": 1_000_000},
        index=idx,
    )


def _trending_down(n: int = 60, slope: float = 0.5) -> pd.DataFrame:
    return _trending_up(n, slope=-slope)


def _flat(n: int = 60) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    close = np.full(n, 100.0)
    return pd.DataFrame(
        {"Close": close, "High": close + 0.2, "Low": close - 0.2, "Volume": 1_000_000},
        index=idx,
    )


def test_indicators_added():
    df = compute_indicators(_trending_up())
    for col in ("EMA_9", "EMA_20", "RSI", "VWAP"):
        assert col in df.columns
    # Final RSI in a steady uptrend should be >> 50
    assert df["RSI"].iloc[-1] > 70


def test_indicators_fallback_without_volume():
    bare = pd.DataFrame({"Close": [100.0 + i for i in range(40)]})
    df = compute_indicators(bare)
    assert "VWAP" in df.columns
    assert not df["VWAP"].isna().all()


def test_call_score_high_in_uptrend():
    df = compute_indicators(_trending_up())
    result = score_price_action(df, "call")
    assert result["trend"] is True
    assert result["momentum"] is True
    assert result["vwap"] is True
    assert result["score"] == pytest.approx(0.45)


def test_put_score_high_in_downtrend():
    df = compute_indicators(_trending_down())
    result = score_price_action(df, "put")
    assert result["trend"] is True
    assert result["momentum"] is True
    assert result["vwap"] is True
    assert result["score"] == pytest.approx(0.45)


def test_call_score_low_in_downtrend():
    df = compute_indicators(_trending_down())
    result = score_price_action(df, "call")
    assert result["score"] == 0.0


def test_signal_fn_fires_in_uptrend():
    signal_fn = make_wizard_signal_fn("call")
    history = compute_indicators(_trending_up())
    assert signal_fn(history) == "buy_call"


def test_signal_fn_holds_in_flat_market():
    # Flat tape: trend fails, momentum sits near 50 — score below threshold
    signal_fn = make_wizard_signal_fn("call")
    history = compute_indicators(_flat())
    assert signal_fn(history) == "hold"


def test_signal_fn_holds_with_too_few_bars():
    signal_fn = make_wizard_signal_fn("call")
    short_history = _trending_up(n=5)
    assert signal_fn(short_history) == "hold"


def test_stretch_filter_suppresses_overextended_calls():
    # Build a series where current close is way above VWAP (steep uptrend).
    df = compute_indicators(_trending_up(n=60, slope=0.6))
    last = df.iloc[-1]
    stretch = (last["Close"] - last["VWAP"]) / last["VWAP"]
    assert stretch > 0.05  # confirm fixture really is overextended

    unfiltered = make_wizard_signal_fn("call")
    filtered = make_wizard_signal_fn("call", max_stretch_pct=0.05)
    assert unfiltered(df) == "buy_call"
    assert filtered(df) == "hold"


def test_stretch_filter_suppresses_overextended_puts():
    df = compute_indicators(_trending_down(n=60, slope=0.6))
    last = df.iloc[-1]
    stretch = (last["Close"] - last["VWAP"]) / last["VWAP"]
    assert stretch < -0.05

    assert make_wizard_signal_fn("put")(df) == "buy_put"
    assert make_wizard_signal_fn("put", max_stretch_pct=0.05)(df) == "hold"


def test_stretch_filter_allows_non_overextended_entries():
    # Mild uptrend — close is above VWAP but not by much.
    df = compute_indicators(_trending_up(n=60, slope=0.05))
    last = df.iloc[-1]
    stretch = (last["Close"] - last["VWAP"]) / last["VWAP"]
    assert 0 < stretch < 0.05  # mild stretch only

    assert make_wizard_signal_fn("call", max_stretch_pct=0.10)(df) == "buy_call"


def test_end_to_end_through_directional_backtester():
    """Wizard's call signal on a sustained uptrend should produce winning trades."""
    history = _trending_up(n=80, slope=0.4)
    signal_fn = make_wizard_signal_fn("call")
    bt = DirectionalBacktester(
        profit_target_pct=0.15, stop_loss_pct=0.08, max_hold_days=10, dte_days=30
    )
    result = bt.run("TEST", "2024-01-01", "2024-04-30", signal_fn, price_history=history)

    assert result.n_trades > 0
    assert result.win_rate > 0.5  # uptrend + bullish signal should win the majority
    assert result.total_pnl > 0
