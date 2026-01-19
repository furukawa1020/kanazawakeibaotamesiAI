"""
Performance metrics for horse racing prediction.
"""
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score, mean_absolute_error
import logging

logger = logging.getLogger(__name__)


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Calculate Brier score for binary predictions.
    
    Args:
        y_true: True binary labels (0 or 1)
        y_prob: Predicted probabilities
    
    Returns:
        Brier score (lower is better)
    """
    return np.mean((y_true - y_prob) ** 2)


def brier_score_multiclass(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Calculate Brier score for multiclass predictions.
    
    Args:
        y_true: True class indices
        y_prob: Predicted probability matrix [n_samples, n_classes]
    
    Returns:
        Brier score
    """
    n_samples, n_classes = y_prob.shape
    
    # Create one-hot encoding of true labels
    y_true_onehot = np.zeros((n_samples, n_classes))
    y_true_onehot[np.arange(n_samples), y_true] = 1
    
    # Calculate Brier score
    return np.mean(np.sum((y_true_onehot - y_prob) ** 2, axis=1))


def calculate_roi(
    stakes: np.ndarray,
    payouts: np.ndarray
) -> float:
    """
    Calculate return on investment (ROI).
    
    Args:
        stakes: Array of stake amounts
        payouts: Array of payout amounts (0 if lost)
    
    Returns:
        ROI as percentage
    """
    total_stake = stakes.sum()
    total_payout = payouts.sum()
    
    if total_stake == 0:
        return 0.0
    
    roi = ((total_payout - total_stake) / total_stake) * 100
    return roi


def calculate_max_drawdown(cumulative_pnl: np.ndarray) -> float:
    """
    Calculate maximum drawdown from cumulative P&L.
    
    Args:
        cumulative_pnl: Array of cumulative profit/loss
    
    Returns:
        Maximum drawdown (positive value)
    """
    if len(cumulative_pnl) == 0:
        return 0.0
    
    # Calculate running maximum
    running_max = np.maximum.accumulate(cumulative_pnl)
    
    # Calculate drawdown at each point
    drawdown = running_max - cumulative_pnl
    
    # Return maximum drawdown
    max_dd = np.max(drawdown)
    return max_dd


def calculate_sharpe_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0
) -> float:
    """
    Calculate Sharpe ratio.
    
    Args:
        returns: Array of returns
        risk_free_rate: Risk-free rate
    
    Returns:
        Sharpe ratio
    """
    if len(returns) == 0:
        return 0.0
    
    excess_returns = returns - risk_free_rate
    
    if np.std(excess_returns) == 0:
        return 0.0
    
    sharpe = np.mean(excess_returns) / np.std(excess_returns)
    return sharpe


def calculate_hit_rate(
    stakes: np.ndarray,
    payouts: np.ndarray
) -> float:
    """
    Calculate hit rate (percentage of winning bets).
    
    Args:
        stakes: Array of stake amounts
        payouts: Array of payout amounts
    
    Returns:
        Hit rate as percentage
    """
    if len(stakes) == 0:
        return 0.0
    
    wins = (payouts > 0).sum()
    hit_rate = (wins / len(stakes)) * 100
    return hit_rate


def calculate_ndcg_at_k(
    y_true: np.ndarray,
    y_score: np.ndarray,
    k: int = 3
) -> float:
    """
    Calculate NDCG@k for ranking evaluation.
    
    Args:
        y_true: True relevance scores (finish positions)
        y_score: Predicted scores
        k: Top-k positions to consider
    
    Returns:
        NDCG@k score
    """
    # Convert finish positions to relevance (1st = highest relevance)
    relevance = 1.0 / y_true
    
    try:
        ndcg = ndcg_score([relevance], [y_score], k=k)
    except:
        ndcg = 0.0
    
    return ndcg


class MetricsCalculator:
    """Calculate comprehensive metrics for model evaluation."""
    
    def __init__(self):
        """Initialize metrics calculator."""
        self.metrics = {}
    
    def calculate_prediction_metrics(
        self,
        df: pd.DataFrame,
        score_col: str = 'score',
        target_col: str = 'finish_position'
    ) -> Dict[str, float]:
        """
        Calculate prediction quality metrics.
        
        Args:
            df: DataFrame with predictions and targets
            score_col: Column name for predicted scores
            target_col: Column name for true finish positions
        
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Group by race
        for race_id, race_df in df.groupby('race_id'):
            scores = race_df[score_col].values
            positions = race_df[target_col].values
            
            # NDCG@k
            for k in [1, 3, 5]:
                ndcg = calculate_ndcg_at_k(positions, scores, k=k)
                metrics.setdefault(f'ndcg@{k}', []).append(ndcg)
        
        # Average metrics across races
        avg_metrics = {
            key: np.mean(values)
            for key, values in metrics.items()
        }
        
        return avg_metrics
    
    def calculate_betting_metrics(
        self,
        bets_df: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Calculate betting performance metrics.
        
        Args:
            bets_df: DataFrame with columns ['stake', 'payout', 'race_id']
        
        Returns:
            Dictionary of metrics
        """
        if len(bets_df) == 0:
            return {
                'total_races': 0,
                'total_bets': 0,
                'total_stake': 0,
                'total_payout': 0,
                'roi': 0,
                'hit_rate': 0,
                'avg_stake_per_race': 0,
                'max_drawdown': 0
            }
        
        stakes = bets_df['stake'].values
        payouts = bets_df['payout'].values
        
        # Calculate cumulative P&L
        pnl = payouts - stakes
        cumulative_pnl = np.cumsum(pnl)
        
        metrics = {
            'total_races': bets_df['race_id'].nunique(),
            'total_bets': len(bets_df),
            'total_stake': int(stakes.sum()),
            'total_payout': int(payouts.sum()),
            'total_profit': int(pnl.sum()),
            'roi': calculate_roi(stakes, payouts),
            'hit_rate': calculate_hit_rate(stakes, payouts),
            'avg_stake_per_race': stakes.sum() / bets_df['race_id'].nunique(),
            'avg_bets_per_race': len(bets_df) / bets_df['race_id'].nunique(),
            'max_drawdown': calculate_max_drawdown(cumulative_pnl),
            'sharpe_ratio': calculate_sharpe_ratio(pnl)
        }
        
        return metrics
