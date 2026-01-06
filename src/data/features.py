"""
Feature engineering for Kanazawa 3T.
CRITICAL: All features must be leak-free (no future information).
"""
from typing import Optional, List
import pandas as pd
import numpy as np
import logging
from datetime import timedelta

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Generate features for horse racing prediction.
    All aggregations are time-aware to prevent data leakage.
    """
    
    def __init__(self, n_past_races: int = 3, min_races_for_stats: int = 2):
        """
        Initialize feature engineer.
        
        Args:
            n_past_races: Number of past races to use for rolling features
            min_races_for_stats: Minimum races needed for aggregated statistics
        """
        self.n_past_races = n_past_races
        self.min_races_for_stats = min_races_for_stats
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all features for the dataset.
        
        Args:
            df: Preprocessed race data (must be sorted by date)
        
        Returns:
            DataFrame with added features
        """
        df = df.copy()
        
        # Verify data is sorted by date
        if not df['date'].is_monotonic_increasing:
            logger.warning("Data not sorted by date. Sorting now...")
            df = df.sort_values(['date', 'race_id', 'horse_no']).reset_index(drop=True)
        
        logger.info("Creating basic features...")
        df = self._create_basic_features(df)
        
        logger.info("Creating horse past performance features...")
        df = self._create_horse_past_features(df)
        
        logger.info("Creating jockey/trainer statistics...")
        df = self._create_jockey_trainer_features(df)
        
        logger.info("Creating same-condition features...")
        df = self._create_same_condition_features(df)
        
        logger.info(f"Feature engineering complete. Total columns: {len(df.columns)}")
        
        return df
    
    def _create_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create basic features from raw data."""
        # Days since last race (if horse_id available)
        if 'horse_id' in df.columns:
            df['days_since_last_race'] = df.groupby('horse_id')['date'].diff().dt.days
            df['days_since_last_race'] = df['days_since_last_race'].fillna(-1)
        
        # Rest category
        if 'days_since_last_race' in df.columns:
            df['rest_category'] = pd.cut(
                df['days_since_last_race'],
                bins=[-np.inf, 0, 14, 30, 60, 90, np.inf],
                labels=['unknown', 'short', 'normal', 'medium', 'long', 'very_long']
            )
            df['rest_category_encoded'] = df['rest_category'].cat.codes
        
        # Weight difference from average (if available)
        if 'weight_carried' in df.columns:
            df['weight_diff_from_avg'] = df.groupby('race_id')['weight_carried'].transform(
                lambda x: x - x.mean()
            )
        
        # Gate position relative to field size
        df['field_size'] = df.groupby('race_id')['horse_no'].transform('count')
        df['gate_position_ratio'] = df['gate'] / df['field_size']
        
        # Distance category
        df['distance_category'] = pd.cut(
            df['distance'],
            bins=[0, 1200, 1600, 2000, 3000],
            labels=['sprint', 'mile', 'intermediate', 'long']
        )
        df['distance_category_encoded'] = df['distance_category'].cat.codes
        
        return df
    
    def _create_horse_past_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features from horse's past performances.
        CRITICAL: Only use data from BEFORE the current race date.
        """
        if 'horse_id' not in df.columns:
            logger.warning("horse_id not available. Skipping horse past features.")
            return df
        
        # For each row, get past N races for that horse
        for i in range(1, self.n_past_races + 1):
            # Shift finish position by i races (within same horse)
            df[f'past_{i}_finish'] = df.groupby('horse_id')['finish_position'].shift(i)
            
            # Shift distance
            df[f'past_{i}_distance'] = df.groupby('horse_id')['distance'].shift(i)
            
            # Shift class
            if 'class_encoded' in df.columns:
                df[f'past_{i}_class'] = df.groupby('horse_id')['class_encoded'].shift(i)
            
            # Shift surface
            if 'surface_encoded' in df.columns:
                df[f'past_{i}_surface'] = df.groupby('horse_id')['surface_encoded'].shift(i)
            
            # Shift track condition
            if 'track_condition_encoded' in df.columns:
                df[f'past_{i}_track_cond'] = df.groupby('horse_id')['track_condition_encoded'].shift(i)
        
        # Fill NaN with -1 (indicating no past data)
        past_cols = [col for col in df.columns if col.startswith('past_')]
        df[past_cols] = df[past_cols].fillna(-1)
        
        # Aggregate statistics from past N races
        df['avg_past_finish'] = df[[f'past_{i}_finish' for i in range(1, self.n_past_races + 1)]].replace(-1, np.nan).mean(axis=1)
        df['best_past_finish'] = df[[f'past_{i}_finish' for i in range(1, self.n_past_races + 1)]].replace(-1, np.nan).min(axis=1)
        df['worst_past_finish'] = df[[f'past_{i}_finish' for i in range(1, self.n_past_races + 1)]].replace(-1, np.nan).max(axis=1)
        
        # Fill NaN with -1
        df['avg_past_finish'] = df['avg_past_finish'].fillna(-1)
        df['best_past_finish'] = df['best_past_finish'].fillna(-1)
        df['worst_past_finish'] = df['worst_past_finish'].fillna(-1)
        
        # Distance change from last race
        if 'past_1_distance' in df.columns:
            df['distance_change'] = df['distance'] - df['past_1_distance']
            df['distance_change'] = df['distance_change'].fillna(0)
        
        # Class change indicator
        if 'past_1_class' in df.columns and 'class_encoded' in df.columns:
            df['class_change'] = df['class_encoded'] - df['past_1_class']
            df['class_change'] = df['class_change'].fillna(0)
        
        return df
    
    def _create_jockey_trainer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create jockey and trainer statistics.
        CRITICAL: Only use data from BEFORE the current race date.
        """
        # For each race, calculate jockey/trainer stats from past data only
        df = df.sort_values(['date', 'race_id']).reset_index(drop=True)
        
        # Jockey win rate (rolling, leak-free)
        if 'jockey_id_encoded' in df.columns:
            df['jockey_win_rate'] = self._calculate_rolling_win_rate(
                df, 'jockey_id_encoded', min_races=self.min_races_for_stats
            )
            df['jockey_top3_rate'] = self._calculate_rolling_top_n_rate(
                df, 'jockey_id_encoded', n=3, min_races=self.min_races_for_stats
            )
        
        # Trainer win rate (rolling, leak-free)
        if 'trainer_id_encoded' in df.columns:
            df['trainer_win_rate'] = self._calculate_rolling_win_rate(
                df, 'trainer_id_encoded', min_races=self.min_races_for_stats
            )
            df['trainer_top3_rate'] = self._calculate_rolling_top_n_rate(
                df, 'trainer_id_encoded', n=3, min_races=self.min_races_for_stats
            )
        
        return df
    
    def _create_same_condition_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features for performance under same conditions.
        CRITICAL: Only use data from BEFORE the current race date.
        """
        if 'horse_id' not in df.columns:
            return df
        
        # Same distance performance (rolling)
        df['same_distance_avg_finish'] = self._calculate_rolling_condition_stat(
            df, 'horse_id', 'distance', 'finish_position', agg='mean'
        )
        
        # Same surface performance (rolling)
        if 'surface_encoded' in df.columns:
            df['same_surface_avg_finish'] = self._calculate_rolling_condition_stat(
                df, 'horse_id', 'surface_encoded', 'finish_position', agg='mean'
            )
        
        # Same class performance (rolling)
        if 'class_encoded' in df.columns:
            df['same_class_avg_finish'] = self._calculate_rolling_condition_stat(
                df, 'horse_id', 'class_encoded', 'finish_position', agg='mean'
            )
        
        return df
    
    def _calculate_rolling_win_rate(
        self,
        df: pd.DataFrame,
        group_col: str,
        min_races: int = 2
    ) -> pd.Series:
        """
        Calculate rolling win rate (leak-free).
        For each row, only use data from BEFORE that date.
        """
        win_rates = []
        
        for idx, row in df.iterrows():
            current_date = row['date']
            group_value = row[group_col]
            
            # Get all past races for this group (before current date)
            past_data = df[(df[group_col] == group_value) & (df['date'] < current_date)]
            
            if len(past_data) < min_races:
                win_rates.append(-1)  # Not enough data
            else:
                win_rate = (past_data['finish_position'] == 1).mean()
                win_rates.append(win_rate)
        
        return pd.Series(win_rates, index=df.index)
    
    def _calculate_rolling_top_n_rate(
        self,
        df: pd.DataFrame,
        group_col: str,
        n: int = 3,
        min_races: int = 2
    ) -> pd.Series:
        """Calculate rolling top-N rate (leak-free)."""
        top_n_rates = []
        
        for idx, row in df.iterrows():
            current_date = row['date']
            group_value = row[group_col]
            
            # Get all past races for this group (before current date)
            past_data = df[(df[group_col] == group_value) & (df['date'] < current_date)]
            
            if len(past_data) < min_races:
                top_n_rates.append(-1)
            else:
                top_n_rate = (past_data['finish_position'] <= n).mean()
                top_n_rates.append(top_n_rate)
        
        return pd.Series(top_n_rates, index=df.index)
    
    def _calculate_rolling_condition_stat(
        self,
        df: pd.DataFrame,
        group_col: str,
        condition_col: str,
        stat_col: str,
        agg: str = 'mean'
    ) -> pd.Series:
        """
        Calculate rolling statistics under same condition (leak-free).
        """
        stats = []
        
        for idx, row in df.iterrows():
            current_date = row['date']
            group_value = row[group_col]
            condition_value = row[condition_col]
            
            # Get past races with same condition
            past_data = df[
                (df[group_col] == group_value) &
                (df[condition_col] == condition_value) &
                (df['date'] < current_date)
            ]
            
            if len(past_data) < self.min_races_for_stats:
                stats.append(-1)
            else:
                if agg == 'mean':
                    stat_value = past_data[stat_col].mean()
                elif agg == 'min':
                    stat_value = past_data[stat_col].min()
                elif agg == 'max':
                    stat_value = past_data[stat_col].max()
                else:
                    stat_value = -1
                stats.append(stat_value)
        
        return pd.Series(stats, index=df.index)
    
    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Get list of feature columns (excluding metadata and target).
        
        Args:
            df: DataFrame with features
        
        Returns:
            List of feature column names
        """
        # Columns to exclude
        exclude_cols = [
            'race_id', 'date', 'horse_id', 'horse_no', 'finish_position',
            'surface', 'track_condition', 'class', 'sex', 'jockey_id', 'trainer_id',
            'time', 'margin', 'closing3f', 'passing_order',
            'distance_category', 'rest_category'  # Categorical versions
        ]
        
        # Get all columns that are not in exclude list
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        return feature_cols
