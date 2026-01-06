"""
Stake allocation for trifecta bets.
"""
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class StakeAllocator:
    """
    Allocate stakes to bets based on probabilities and budget.
    """
    
    def __init__(
        self,
        budget: int = 3000,
        stake_unit: int = 100,
        min_stake: int = 100,
        probability_exponent: float = 0.7
    ):
        """
        Initialize allocator.
        
        Args:
            budget: Total budget per race (yen)
            stake_unit: Minimum stake unit (yen)
            min_stake: Minimum stake per bet (yen)
            probability_exponent: Exponent for probability weighting (q = p^α)
        """
        self.budget = budget
        self.stake_unit = stake_unit
        self.min_stake = min_stake
        self.probability_exponent = probability_exponent
    
    def allocate(self, bets_df: pd.DataFrame) -> pd.DataFrame:
        """
        Allocate stakes to bets.
        
        Args:
            bets_df: DataFrame with columns ['first', 'second', 'third', 'probability', 'odds', 'ev']
        
        Returns:
            DataFrame with added 'stake' column
        """
        if len(bets_df) == 0:
            logger.warning("No bets to allocate stakes to")
            return bets_df
        
        # Calculate weights (q = p^α)
        weights = bets_df['probability'].values ** self.probability_exponent
        
        # Normalize weights
        weights_norm = weights / weights.sum()
        
        # Calculate raw stakes
        raw_stakes = weights_norm * self.budget
        
        # Round to stake unit
        stakes = np.round(raw_stakes / self.stake_unit) * self.stake_unit
        
        # Ensure minimum stake
        stakes = np.maximum(stakes, self.min_stake)
        
        # Adjust to fit budget
        stakes = self._adjust_to_budget(stakes, bets_df['ev'].values)
        
        # Add to dataframe
        result_df = bets_df.copy()
        result_df['stake'] = stakes.astype(int)
        
        # Remove bets with 0 stake
        result_df = result_df[result_df['stake'] > 0].reset_index(drop=True)
        
        total_stake = result_df['stake'].sum()
        logger.info(f"Allocated {total_stake} yen across {len(result_df)} bets (budget: {self.budget})")
        
        return result_df
    
    def _adjust_to_budget(self, stakes: np.ndarray, evs: np.ndarray) -> np.ndarray:
        """
        Adjust stakes to fit within budget.
        
        Args:
            stakes: Array of stakes
            evs: Array of expected values
        
        Returns:
            Adjusted stakes
        """
        total = stakes.sum()
        
        if total <= self.budget:
            return stakes
        
        # Need to reduce stakes
        # Strategy: reduce proportionally, prioritizing lower EV bets
        
        # Sort by EV (ascending)
        sorted_indices = np.argsort(evs)
        
        adjusted_stakes = stakes.copy()
        
        # Iteratively reduce stakes from lowest EV bets
        for idx in sorted_indices:
            if adjusted_stakes.sum() <= self.budget:
                break
            
            # Reduce this bet's stake
            reduction = min(self.stake_unit, adjusted_stakes[idx] - self.min_stake)
            if reduction > 0:
                adjusted_stakes[idx] -= reduction
        
        # If still over budget, remove lowest EV bets
        while adjusted_stakes.sum() > self.budget:
            # Find bet with lowest EV that has stake > 0
            valid_mask = adjusted_stakes > 0
            if not valid_mask.any():
                break
            
            valid_evs = np.where(valid_mask, evs, np.inf)
            idx_to_remove = np.argmin(valid_evs)
            adjusted_stakes[idx_to_remove] = 0
        
        return adjusted_stakes
    
    def calculate_expected_return(self, bets_df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate expected return statistics.
        
        Args:
            bets_df: DataFrame with stakes allocated
        
        Returns:
            Dictionary of statistics
        """
        if len(bets_df) == 0 or 'stake' not in bets_df.columns:
            return {
                'total_stake': 0,
                'expected_return': 0,
                'expected_profit': 0,
                'roi': 0
            }
        
        total_stake = bets_df['stake'].sum()
        
        # Expected return for each bet
        expected_returns = bets_df['probability'] * bets_df['odds'] * bets_df['stake']
        total_expected_return = expected_returns.sum()
        
        expected_profit = total_expected_return - total_stake
        roi = (expected_profit / total_stake * 100) if total_stake > 0 else 0
        
        return {
            'total_stake': int(total_stake),
            'expected_return': float(total_expected_return),
            'expected_profit': float(expected_profit),
            'roi': float(roi)
        }
    
    def format_bets_for_output(self, bets_df: pd.DataFrame) -> pd.DataFrame:
        """
        Format bets for output (CSV/JSON).
        
        Args:
            bets_df: DataFrame with all bet information
        
        Returns:
            Formatted DataFrame
        """
        if len(bets_df) == 0:
            return pd.DataFrame()
        
        # Create ticket string (e.g., "1-2-3")
        bets_df['ticket'] = (
            bets_df['first'].astype(str) + '-' +
            bets_df['second'].astype(str) + '-' +
            bets_df['third'].astype(str)
        )
        
        # Select and order columns
        output_cols = ['ticket', 'probability', 'odds', 'ev', 'stake']
        output_df = bets_df[output_cols].copy()
        
        # Round for readability
        output_df['probability'] = output_df['probability'].round(4)
        output_df['odds'] = output_df['odds'].round(1)
        output_df['ev'] = output_df['ev'].round(3)
        
        return output_df
