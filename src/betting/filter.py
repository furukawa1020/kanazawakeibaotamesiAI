"""
EV-based filtering for bet candidates.
"""
from typing import List, Tuple, Dict, Optional
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class BetFilter:
    """
    Filter bet candidates based on expected value (EV) and other criteria.
    """
    
    def __init__(
        self,
        min_ev: float = 1.15,
        min_probability: float = 0.002,
        max_bets: int = 30
    ):
        """
        Initialize filter.
        
        Args:
            min_ev: Minimum expected value threshold
            min_probability: Minimum probability threshold
            max_bets: Maximum number of bets to return
        """
        self.min_ev = min_ev
        self.min_probability = min_probability
        self.max_bets = max_bets
    
    def filter_candidates(
        self,
        candidates: List[Tuple[int, int, int]],
        probabilities: Dict[Tuple[int, int, int], float],
        odds: Dict[Tuple[int, int, int], float]
    ) -> pd.DataFrame:
        """
        Filter candidates and calculate EV.
        
        Args:
            candidates: List of trifecta candidates
            probabilities: Dictionary of probabilities for each candidate
            odds: Dictionary of odds for each candidate
        
        Returns:
            DataFrame with filtered candidates and their EV
        """
        results = []
        
        for combo in candidates:
            # Get probability and odds
            prob = probabilities.get(combo, 0.0)
            odd = odds.get(combo)
            
            if odd is None:
                continue  # Skip if odds not available
            
            # Calculate EV
            ev = prob * odd
            
            # Apply filters
            if prob < self.min_probability:
                continue
            
            if ev < self.min_ev:
                continue
            
            results.append({
                'first': combo[0],
                'second': combo[1],
                'third': combo[2],
                'probability': prob,
                'odds': odd,
                'ev': ev
            })
        
        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        if len(df) == 0:
            logger.warning("No candidates passed filters")
            return df
        
        # Sort by EV (descending)
        df = df.sort_values('ev', ascending=False)
        
        # Limit to max_bets
        if len(df) > self.max_bets:
            logger.info(f"Limiting {len(df)} candidates to {self.max_bets} bets")
            df = df.head(self.max_bets)
        
        logger.info(f"Filtered to {len(df)} bets. EV range: [{df['ev'].min():.3f}, {df['ev'].max():.3f}]")
        
        return df.reset_index(drop=True)
    
    def get_statistics(self, filtered_df: pd.DataFrame) -> Dict[str, float]:
        """
        Get statistics about filtered bets.
        
        Args:
            filtered_df: DataFrame of filtered bets
        
        Returns:
            Dictionary of statistics
        """
        if len(filtered_df) == 0:
            return {
                'n_bets': 0,
                'mean_ev': 0.0,
                'median_ev': 0.0,
                'mean_prob': 0.0,
                'total_prob': 0.0
            }
        
        return {
            'n_bets': len(filtered_df),
            'mean_ev': filtered_df['ev'].mean(),
            'median_ev': filtered_df['ev'].median(),
            'max_ev': filtered_df['ev'].max(),
            'min_ev': filtered_df['ev'].min(),
            'mean_prob': filtered_df['probability'].mean(),
            'total_prob': filtered_df['probability'].sum(),
            'mean_odds': filtered_df['odds'].mean()
        }
