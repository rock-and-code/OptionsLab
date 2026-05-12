import numpy as np
import pandas as pd
import pytest

from src.backtesting import BacktestResult, DirectionalBacktester, rolling_realized_vol


def _make_history(prices: list[float]) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=len(prices), freq="B")
    return pd.DataFrame({"Close": prices}, index=idx)


def _buy_on_first_bar(history: pd.DataFrame) -> str:
    return "buy_call" if len(history) == 1 else "hold"


def _buy_put_on_first_bar(history: pd.DataFrame) -> str:
    return "buy_put" if len(history) == 1 else "hold"


def _never_buy(history: pd.DataFrame) -> str:
    return "hold"


def test_returns_backtest_result_shape():
    hist = _make_history([100.0] * 5)
    bt = DirectionalBacktester()
    result = bt.run("TEST", "2024-01-01", "2024-01-08", _never_buy, price_history=hist)

    assert isinstance(result, BacktestResult)
    assert result.n_trades == 0
    assert result.total_pnl == 0.0
    assert result.win_rate == 0.0


def test_call_profits_when_underlying_rallies():
    # Sharp rally — ATM call bought day 1 should hit profit target.
    prices = [100.0] + [100.0 + i for i in range(1, 15)]
    hist = _make_history(prices)
    bt = DirectionalBacktester(profit_target_pct=0.15, stop_loss_pct=0.50)
    result = bt.run("TEST", "2024-01-01", "2024-01-22", _buy_on_first_bar, price_history=hist)

    assert result.n_trades == 1
    assert result.total_pnl > 0
    assert result.win_rate == 1.0


def test_call_loses_when_underlying_sells_off():
    # Sharp drop — ATM call should hit stop loss.
    prices = [100.0] + [100.0 - i for i in range(1, 15)]
    hist = _make_history(prices)
    bt = DirectionalBacktester(profit_target_pct=0.50, stop_loss_pct=0.08)
    result = bt.run("TEST", "2024-01-01", "2024-01-22", _buy_on_first_bar, price_history=hist)

    assert result.n_trades == 1
    assert result.total_pnl < 0
    assert result.win_rate == 0.0


def test_put_profits_when_underlying_sells_off():
    prices = [100.0] + [100.0 - i for i in range(1, 15)]
    hist = _make_history(prices)
    bt = DirectionalBacktester(profit_target_pct=0.15, stop_loss_pct=0.50)
    result = bt.run("TEST", "2024-01-01", "2024-01-22", _buy_put_on_first_bar, price_history=hist)

    assert result.n_trades == 1
    assert result.total_pnl > 0


def test_max_hold_forces_exit_in_flat_market():
    # Flat tape — neither stop nor target hits; max_hold should close it.
    prices = [100.0] * 20
    hist = _make_history(prices)
    bt = DirectionalBacktester(max_hold_days=5, dte_days=30)
    result = bt.run("TEST", "2024-01-01", "2024-01-29", _buy_on_first_bar, price_history=hist)

    assert result.n_trades == 1
    # Theta decay on a flat tape -> small loss expected
    assert result.total_pnl < 0


def test_signal_ignored_while_position_open():
    # Always-call signal — a new trade may open every bar that has no open position,
    # but never two open positions at once. With min 1 bar between exit and re-entry,
    # n_trades is bounded by n_bars / 2.
    def always_call(history: pd.DataFrame) -> str:
        return "buy_call"

    prices = [100.0 + np.sin(i / 3) for i in range(40)]
    hist = _make_history(prices)
    bt = DirectionalBacktester(max_hold_days=5)
    result = bt.run("TEST", "2024-01-01", "2024-02-29", always_call, price_history=hist)

    assert result.n_trades > 1  # multiple sequential trades did fire
    assert result.n_trades <= len(prices) // 2  # never overlapping


def test_rejects_too_short_history():
    hist = _make_history([100.0])
    bt = DirectionalBacktester()
    with pytest.raises(ValueError):
        bt.run("TEST", "2024-01-01", "2024-01-02", _buy_on_first_bar, price_history=hist)


def test_rolling_realized_vol_shape_and_warmup():
    closes = pd.Series(100.0 + np.cumsum(np.random.RandomState(0).randn(50) * 0.5))
    vol = rolling_realized_vol(closes, window=20)
    assert len(vol) == len(closes)
    assert vol.iloc[:20].isna().all()  # warm-up period
    assert not vol.iloc[20:].isna().any()
    assert (vol.iloc[20:] > 0).all()


def test_sigma_series_locks_at_entry_and_changes_pricing():
    """Entry sigma should be drawn from sigma_series; higher entry sigma -> higher premium."""
    prices = [100.0] + [100.0 + i * 0.1 for i in range(1, 30)]  # mild drift
    hist = _make_history(prices)

    low_sigma = pd.Series([0.10] * len(prices), index=hist.index)
    high_sigma = pd.Series([0.40] * len(prices), index=hist.index)

    bt = DirectionalBacktester(profit_target_pct=0.50, stop_loss_pct=0.50, max_hold_days=5)

    r_low = bt.run("T", "2024-01-01", "2024-02-01", _buy_on_first_bar,
                   price_history=hist, sigma_series=low_sigma)
    r_high = bt.run("T", "2024-01-01", "2024-02-01", _buy_on_first_bar,
                    price_history=hist, sigma_series=high_sigma)

    # Same path, same signal, same target/stop — but the higher-sigma run has a
    # higher entry premium, so a fixed absolute spot move clears a smaller pct.
    assert r_low.n_trades == 1 and r_high.n_trades == 1
    assert r_low.total_pnl != r_high.total_pnl


def test_sigma_series_nan_falls_back_to_default():
    prices = [100.0 + i * 0.5 for i in range(20)]
    hist = _make_history(prices)
    sigma_series = pd.Series([np.nan] * len(prices), index=hist.index)
    bt = DirectionalBacktester(sigma=0.20, max_hold_days=5)
    # Should run without error even though every sigma is NaN.
    result = bt.run("T", "2024-01-01", "2024-01-29", _buy_on_first_bar,
                    price_history=hist, sigma_series=sigma_series)
    assert result.n_trades == 1


def test_sigma_scaled_targets_widens_bounds_at_high_vol():
    """High-sigma entries should get wider effective targets/stops."""
    # Mild moves so we rely on theta and time rather than spot crossing the bounds.
    prices = [100.0 + 0.05 * i for i in range(30)]
    hist = _make_history(prices)

    # Lock sigma high so the position enters with sigma_at_entry = 0.50
    high_sigma = pd.Series([0.50] * len(prices), index=hist.index)

    # Without scaling: fixed 8% stop, fixed 15% target.
    unscaled = DirectionalBacktester(
        profit_target_pct=0.15, stop_loss_pct=0.08, max_hold_days=10,
        sigma_scaled_targets=False,
    )
    r_unscaled = unscaled.run(
        "T", "2024-01-01", "2024-02-12", _buy_on_first_bar,
        price_history=hist, sigma_series=high_sigma,
    )

    # With scaling at reference 0.20: effective stop = 8% * 0.50/0.20 = 20%,
    # effective target = 15% * 0.50/0.20 = 37.5%. Same path -> exits should
    # differ; in particular the scaled run will *not* trigger the unscaled
    # stop level on small adverse moves.
    scaled = DirectionalBacktester(
        profit_target_pct=0.15, stop_loss_pct=0.08, max_hold_days=10,
        sigma_scaled_targets=True, reference_sigma=0.20,
    )
    r_scaled = scaled.run(
        "T", "2024-01-01", "2024-02-12", _buy_on_first_bar,
        price_history=hist, sigma_series=high_sigma,
    )

    assert r_unscaled.n_trades >= 1
    assert r_scaled.n_trades >= 1
    # Different effective bounds -> different P&L paths on the same input.
    assert r_unscaled.total_pnl != r_scaled.total_pnl


def test_sigma_scaling_collapses_to_unscaled_at_reference_sigma():
    """When sigma_at_entry == reference_sigma, the scale factor is 1 and
    scaled mode should be identical to unscaled."""
    prices = [100.0 + 0.4 * i for i in range(30)]
    hist = _make_history(prices)
    sigma = pd.Series([0.20] * len(prices), index=hist.index)

    common = dict(
        profit_target_pct=0.15, stop_loss_pct=0.08, max_hold_days=10, dte_days=30,
    )
    a = DirectionalBacktester(**common, sigma_scaled_targets=False)
    b = DirectionalBacktester(**common, sigma_scaled_targets=True, reference_sigma=0.20)

    ra = a.run("T", "2024-01-01", "2024-02-12", _buy_on_first_bar,
               price_history=hist, sigma_series=sigma)
    rb = b.run("T", "2024-01-01", "2024-02-12", _buy_on_first_bar,
               price_history=hist, sigma_series=sigma)

    assert ra.total_pnl == pytest.approx(rb.total_pnl, abs=1e-9)
    assert ra.n_trades == rb.n_trades


def test_sigma_series_wrong_length_raises():
    hist = _make_history([100.0] * 10)
    bt = DirectionalBacktester()
    with pytest.raises(ValueError, match="sigma_series length"):
        bt.run("T", "2024-01-01", "2024-01-15", _buy_on_first_bar,
               price_history=hist, sigma_series=pd.Series([0.2] * 5))
