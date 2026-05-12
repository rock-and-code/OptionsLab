# src/strategies/__init__.py
"""Trading strategies that produce signals consumable by the backtesters."""

from src.strategies.wizard_signal import (
    compute_indicators,
    make_wizard_signal_fn,
    score_price_action,
)

__all__ = [
    "compute_indicators",
    "make_wizard_signal_fn",
    "score_price_action",
]
