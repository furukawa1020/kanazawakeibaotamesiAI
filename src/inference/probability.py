"""
Probability estimation using Plackett-Luce model.
"""
from typing import Dict, Tuple, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)


class ProbabilityEstimator:
    """
    Estimate probabilities for race outcomes using Plackett-Luce model.
    """
    
    def __init__(self, method: str = 'monte_carlo'):
        """
        Initialize probability estimator.
        
        Args:
            method: Estimation method ('monte_carlo' or 'analytical')
        """
        self.method = method
    
    def estimate_win_probabilities(self, weights: np.ndarray) -> np.ndarray:
        """
        Estimate win probability for each horse.
        
        Args:
            weights: Array of weights for each horse
        
        Returns:
            Array of win probabilities (sums to 1.0)
        """
        # In Plackett-Luce, win probability is simply normalized weight
        probs = weights / weights.sum()
        return probs
    
    def estimate_top_n_probabilities(
        self,
        weights: np.ndarray,
        n: int = 3
    ) -> np.ndarray:
        """
        Estimate probability of finishing in top-N for each horse.
        
        This is an approximation. For exact values, use Monte Carlo sampling.
        
        Args:
            weights: Array of weights for each horse
            n: Top-N threshold
        
        Returns:
            Array of top-N probabilities
        """
        # Simple approximation: use softmax-like transformation
        # More accurate estimation requires Monte Carlo
        n_horses = len(weights)
        
        if n >= n_horses:
            return np.ones(n_horses)
        
        # Normalize weights
        w_norm = weights / weights.sum()
        
        # Approximate top-N probability
        # Higher weight = higher chance of top-N
        # This is a heuristic, not exact
        top_n_probs = np.minimum(w_norm * n * 1.5, 1.0)
        
        return top_n_probs
    
    def combine_trifecta_probabilities(
        self,
        trifecta_probs: Dict[Tuple[int, int, int], float],
        min_prob: float = 0.0
    ) -> Dict[Tuple[int, int, int], float]:
        """
        Filter and normalize trifecta probabilities.
        
        Args:
            trifecta_probs: Dictionary of trifecta probabilities
            min_prob: Minimum probability threshold
        
        Returns:
            Filtered and normalized probabilities
        """
        # Filter by minimum probability
        filtered = {
            combo: prob
            for combo, prob in trifecta_probs.items()
            if prob >= min_prob
        }
        
        # Normalize
        total = sum(filtered.values())
        if total > 0:
            normalized = {
                combo: prob / total
                for combo, prob in filtered.items()
            }
        else:
            normalized = filtered
        
        return normalized
    
    def get_top_combinations(
        self,
        trifecta_probs: Dict[Tuple[int, int, int], float],
        top_k: int = 30
    ) -> Dict[Tuple[int, int, int], float]:
        """
        Get top-K trifecta combinations by probability.
        
        Args:
            trifecta_probs: Dictionary of trifecta probabilities
            top_k: Number of top combinations to return
        
        Returns:
            Dictionary of top-K combinations
        """
        # Sort by probability
        sorted_combos = sorted(
            trifecta_probs.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Take top-K
        top_combos = dict(sorted_combos[:top_k])
        
        return top_combos
