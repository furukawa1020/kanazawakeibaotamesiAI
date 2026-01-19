"""
Baseline strategies for comparison.
"""
from typing import List, Tuple, Dict
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class BaselineStrategy:
    """Base class for baseline strategies."""
    
    def generate_bets(
        self,
        race_df: pd.DataFrame,
        odds_df: pd.DataFrame,
        max_bets: int = 30
    ) -> pd.DataFrame:
        """
        Generate bets for a race.
        
        Args:
            race_df: DataFrame with race data
            odds_df: DataFrame with trifecta odds
            max_bets: Maximum number of bets
        
        Returns:
            DataFrame with bet recommendations
        """
        raise NotImplementedError


class PopularityBaseline(BaselineStrategy):
    """
    Baseline strategy based on betting odds (popularity).
    Assumes lower odds = more popular = more likely to win.
    """
    
    def __init__(self, top_k: int = 5):
        """
        Initialize popularity baseline.
        
        Args:
            top_k: Number of top popular horses to consider
        """
        self.top_k = top_k
    
    def generate_bets(
        self,
        race_df: pd.DataFrame,
        odds_df: pd.DataFrame,
        max_bets: int = 30
    ) -> pd.DataFrame:
        """Generate bets based on popularity (win odds)."""
        # This requires win odds data
        # For simplicity, we'll use a proxy: assume horses with lower numbers are more popular
        # In real implementation, use actual win odds
        
        n_horses = len(race_df)
        top_horses = list(range(min(self.top_k, n_horses)))
        
        # Generate all permutations of top horses
        from itertools import permutations
        candidates = list(permutations(top_horses, 3))
        
        # Filter by available odds
        available_combos = set(
            zip(odds_df['first'], odds_df['second'], odds_df['third'])
        )
        
        valid_candidates = [c for c in candidates if c in available_combos]
        
        # Take top max_bets by odds (lower odds = higher priority)
        bets = []
        for combo in valid_candidates[:max_bets]:
            odds_row = odds_df[
                (odds_df['first'] == combo[0]) &
                (odds_df['second'] == combo[1]) &
                (odds_df['third'] == combo[2])
            ]
            
            if len(odds_row) > 0:
                bets.append({
                    'first': combo[0],
                    'second': combo[1],
                    'third': combo[2],
                    'odds': odds_row.iloc[0]['odds'],
                    'stake': 100  # Fixed stake
                })
        
        return pd.DataFrame(bets)


class LastRaceBaseline(BaselineStrategy):
    """
    Baseline strategy based on last race finish position.
    """
    
    def __init__(self, top_k: int = 5):
        """
        Initialize last race baseline.
        
        Args:
            top_k: Number of top horses by last race to consider
        """
        self.top_k = top_k
    
    def generate_bets(
        self,
        race_df: pd.DataFrame,
        odds_df: pd.DataFrame,
        max_bets: int = 30
    ) -> pd.DataFrame:
        """Generate bets based on last race performance."""
        # Get horses sorted by last race finish position
        if 'past_1_finish' not in race_df.columns:
            logger.warning("No past race data available. Using random selection.")
            top_horses = list(range(min(self.top_k, len(race_df))))
        else:
            # Sort by last race finish (lower is better)
            race_df_sorted = race_df.sort_values('past_1_finish')
            top_horses = race_df_sorted.head(self.top_k)['horse_no'].tolist()
        
        # Generate permutations
        from itertools import permutations
        candidates = list(permutations(top_horses, 3))
        
        # Filter by available odds
        available_combos = set(
            zip(odds_df['first'], odds_df['second'], odds_df['third'])
        )
        
        valid_candidates = [c for c in candidates if c in available_combos]
        
        # Create bets
        bets = []
        for combo in valid_candidates[:max_bets]:
            odds_row = odds_df[
                (odds_df['first'] == combo[0]) &
                (odds_df['second'] == combo[1]) &
                (odds_df['third'] == combo[2])
            ]
            
            if len(odds_row) > 0:
                bets.append({
                    'first': combo[0],
                    'second': combo[1],
                    'third': combo[2],
                    'odds': odds_row.iloc[0]['odds'],
                    'stake': 100
                })
        
        return pd.DataFrame(bets)


class JockeyBaseline(BaselineStrategy):
    """
    Baseline strategy based on jockey win rate.
    """
    
    def __init__(self, top_k: int = 5):
        """
        Initialize jockey baseline.
        
        Args:
            top_k: Number of top horses by jockey to consider
        """
        self.top_k = top_k
    
    def generate_bets(
        self,
        race_df: pd.DataFrame,
        odds_df: pd.DataFrame,
        max_bets: int = 30
    ) -> pd.DataFrame:
        """Generate bets based on jockey win rate."""
        # Get horses sorted by jockey win rate
        if 'jockey_win_rate' not in race_df.columns:
            logger.warning("No jockey win rate data. Using random selection.")
            top_horses = list(range(min(self.top_k, len(race_df))))
        else:
            # Sort by jockey win rate (higher is better)
            race_df_sorted = race_df.sort_values('jockey_win_rate', ascending=False)
            top_horses = race_df_sorted.head(self.top_k)['horse_no'].tolist()
        
        # Generate permutations
        from itertools import permutations
        candidates = list(permutations(top_horses, 3))
        
        # Filter by available odds
        available_combos = set(
            zip(odds_df['first'], odds_df['second'], odds_df['third'])
        )
        
        valid_candidates = [c for c in candidates if c in available_combos]
        
        # Create bets
        bets = []
        for combo in valid_candidates[:max_bets]:
            odds_row = odds_df[
                (odds_df['first'] == combo[0]) &
                (odds_df['second'] == combo[1]) &
                (odds_df['third'] == combo[2])
            ]
            
            if len(odds_row) > 0:
                bets.append({
                    'first': combo[0],
                    'second': combo[1],
                    'third': combo[2],
                    'odds': odds_row.iloc[0]['odds'],
                    'stake': 100
                })
        
        return pd.DataFrame(bets)


def get_baseline_strategy(name: str) -> BaselineStrategy:
    """
    Get baseline strategy by name.
    
    Args:
        name: Strategy name ('popularity', 'last_race', 'jockey')
    
    Returns:
        Baseline strategy instance
    """
    strategies = {
        'popularity': PopularityBaseline,
        'last_race': LastRaceBaseline,
        'jockey': JockeyBaseline
    }
    
    if name not in strategies:
        raise ValueError(f"Unknown baseline strategy: {name}")
    
    return strategies[name]()
