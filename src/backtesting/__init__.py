# src/backtesting/__init__.py
"""
Backtesting module for options strategies.

Provides historical P&L simulation and strategy validation.
"""

from src.backtesting.backtest_engine import (
    BacktestEngine,
    BacktestResult,
    run_delta_hedge_backtest,
)
from src.backtesting.directional_engine import (
    DirectionalBacktester,
    rolling_realized_vol,
)

__all__ = [
    "BacktestEngine",
    "BacktestResult",
    "DirectionalBacktester",
    "rolling_realized_vol",
    "run_delta_hedge_backtest",
]
