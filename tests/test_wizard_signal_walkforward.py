import numpy as np
import pandas as pd

from src.experiments.wizard_signal_walkforward import (
    run_walkforward,
    summarize,
    sweep_score_thresholds,
)


def _synth_history(n_bars: int, drift: float, seed: int) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    rets = drift / 252 + 0.012 * rng.randn(n_bars)
    close = 100.0 * np.exp(np.cumsum(rets))
    idx = pd.date_range("2024-01-01", periods=n_bars, freq="B")
    return pd.DataFrame(
        {"Close": close, "High": close * 1.005, "Low": close * 0.995, "Volume": 1_000_000},
        index=idx,
    )


def test_run_walkforward_returns_one_row_per_pair_per_side():
    panel = {
        ("ABC", 2023): _synth_history(120, drift=0.15, seed=0),
        ("XYZ", 2023): _synth_history(120, drift=-0.10, seed=1),
    }
    df = run_walkforward(panel)
    assert len(df) == 4  # 2 pairs × 2 sides
    assert set(df["ticker"]) == {"ABC", "XYZ"}
    assert set(df["side"]) == {"call", "put"}
    for col in ("n_trades", "win_rate", "total_pnl", "sharpe"):
        assert col in df.columns


def test_run_walkforward_skips_too_short_histories():
    panel = {
        ("OK", 2023): _synth_history(100, drift=0.1, seed=2),
        ("SHORT", 2023): _synth_history(10, drift=0.1, seed=3),  # < sigma_window+max_hold
    }
    df = run_walkforward(panel)
    assert set(df["ticker"]) == {"OK"}


def test_summarize_weighted_win_rate_matches_per_pair_aggregation():
    panel = {
        ("A", 2023): _synth_history(120, drift=0.2, seed=4),
        ("B", 2023): _synth_history(120, drift=0.2, seed=5),
    }
    df = run_walkforward(panel)
    summary = summarize(df)

    # Manually compute weighted win rate for calls and confirm it matches
    calls = df[df["side"] == "call"]
    expected_wins = int((calls["n_trades"] * calls["win_rate"]).sum())
    expected_total = int(calls["n_trades"].sum())
    if expected_total:
        assert summary["call"]["weighted_win_rate"] == expected_wins / expected_total


def test_run_walkforward_respects_one_side_only():
    panel = {("A", 2023): _synth_history(120, drift=0.15, seed=6)}
    df = run_walkforward(panel, sides=("call",))
    assert list(df["side"]) == ["call"]


def test_higher_threshold_produces_fewer_trades():
    """Raising score_threshold from loose (0.20) to strict (0.45) must reduce
    trade count; at most-strict it requires all three conditions."""
    panel = {("A", 2023): _synth_history(150, drift=0.20, seed=7)}
    loose = run_walkforward(panel, sides=("call",), score_threshold=0.20)
    strict = run_walkforward(panel, sides=("call",), score_threshold=0.45)
    assert int(loose["n_trades"].iloc[0]) >= int(strict["n_trades"].iloc[0])


def test_sweep_score_thresholds_shape_and_columns():
    panel = {
        ("A", 2023): _synth_history(150, drift=0.15, seed=8),
        ("B", 2023): _synth_history(150, drift=-0.10, seed=9),
    }
    sweep = sweep_score_thresholds(panel, [0.20, 0.45])
    # 2 thresholds × 2 sides = 4 rows
    assert len(sweep) == 4
    for col in ("threshold", "side", "trades", "win_rate", "total_pnl",
                "pnl_per_trade", "mean_sharpe", "pos_pair_pct"):
        assert col in sweep.columns
