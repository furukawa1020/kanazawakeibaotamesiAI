"""
Trifecta bet candidate generation.
"""
from typing import List, Tuple, Set, Optional
import numpy as np
import logging
from itertools import permutations

logger = logging.getLogger(__name__)


class TrifectaGenerator:
    """
    Generate trifecta bet candidates.
    """
    
    def __init__(
        self,
        top_k_horses: int = 7,
        method: str = 'sampling'
    ):
        """
        Initialize generator.
        
        Args:
            top_k_horses: Number of top horses to consider
            method: Generation method ('sampling' or 'enumerate')
        """
        self.top_k_horses = top_k_horses
        self.method = method
    
    def generate_candidates(
        self,
        weights: np.ndarray,
        sampled_probs: Optional[dict] = None,
        min_candidates: int = 10
    ) -> List[Tuple[int, int, int]]:
        """
        Generate trifecta candidates.
        
        Args:
            weights: Array of weights for each horse
            sampled_probs: Optional dictionary of sampled probabilities
            min_candidates: Minimum number of candidates to generate
        
        Returns:
            List of (1st, 2nd, 3rd) tuples
        """
        n_horses = len(weights)
        
        if n_horses < 3:
            raise ValueError("Need at least 3 horses for trifecta")
        
        if self.method == 'enumerate':
            return self._enumerate_candidates(weights)
        elif self.method == 'sampling':
            return self._sampling_candidates(weights, sampled_probs, min_candidates)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _enumerate_candidates(self, weights: np.ndarray) -> List[Tuple[int, int, int]]:
        """
        Enumerate all possible trifecta combinations from top-K horses.
        
        Args:
            weights: Array of weights
        
        Returns:
            List of all permutations
        """
        # Get top-K horses by weight
        top_k = min(self.top_k_horses, len(weights))
        top_indices = np.argsort(weights)[-top_k:][::-1]
        
        # Generate all permutations of length 3
        candidates = list(permutations(top_indices, 3))
        
        logger.info(f"Enumerated {len(candidates)} candidates from top-{top_k} horses")
        
        return candidates
    
    def _sampling_candidates(
        self,
        weights: np.ndarray,
        sampled_probs: Optional[dict],
        min_candidates: int
    ) -> List[Tuple[int, int, int]]:
        """
        Generate candidates from sampling results.
        
        Args:
            weights: Array of weights
            sampled_probs: Dictionary of sampled probabilities
            min_candidates: Minimum number of candidates
        
        Returns:
            List of top candidates by sampled probability
        """
        if sampled_probs is None or len(sampled_probs) == 0:
            # Fall back to enumeration
            logger.warning("No sampled probabilities provided. Falling back to enumeration.")
            return self._enumerate_candidates(weights)
        
        # Sort by probability and take top candidates
        sorted_candidates = sorted(
            sampled_probs.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Take at least min_candidates
        n_candidates = max(min_candidates, len(sorted_candidates) // 10)
        candidates = [combo for combo, prob in sorted_candidates[:n_candidates]]
        
        logger.info(f"Generated {len(candidates)} candidates from sampling")
        
        return candidates
    
    def filter_by_odds_availability(
        self,
        candidates: List[Tuple[int, int, int]],
        available_odds: Set[Tuple[int, int, int]]
    ) -> List[Tuple[int, int, int]]:
        """
        Filter candidates to only those with available odds.
        
        Args:
            candidates: List of candidate trifectas
            available_odds: Set of trifectas with available odds
        
        Returns:
            Filtered list of candidates
        """
        filtered = [c for c in candidates if c in available_odds]
        
        logger.info(f"Filtered {len(candidates)} -> {len(filtered)} candidates by odds availability")
        
        return filtered
