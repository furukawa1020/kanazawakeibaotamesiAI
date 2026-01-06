"""Evaluation package."""
from .metrics import MetricsCalculator, brier_score, calculate_roi, calculate_max_drawdown
from .baselines import get_baseline_strategy, PopularityBaseline, LastRaceBaseline, JockeyBaseline
from .backtest import Backtester

__all__ = [
    'MetricsCalculator',
    'brier_score',
    'calculate_roi',
    'calculate_max_drawdown',
    'get_baseline_strategy',
    'PopularityBaseline',
    'LastRaceBaseline',
    'JockeyBaseline',
    'Backtester'
]
