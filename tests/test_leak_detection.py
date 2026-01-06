"""
Unit tests for leak detection in feature engineering.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data import FeatureEngineer, DataPreprocessor


def create_sample_data():
    """Create sample race data for testing."""
    dates = [datetime(2023, 1, 1) + timedelta(days=i*7) for i in range(10)]
    
    data = []
    for race_idx, date in enumerate(dates):
        for horse_no in range(1, 6):  # 5 horses per race
            data.append({
                'race_id': f'race_{race_idx}',
                'date': date,
                'horse_id': f'horse_{horse_no}',
                'horse_no': horse_no,
                'distance': 1600,
                'surface': 'ダ',
                'track_condition': '良',
                'class': 'B',
                'gate': horse_no,
                'sex': 'M',
                'age': 4,
                'weight_carried': 54.0,
                'jockey_id': f'jockey_{horse_no}',
                'trainer_id': f'trainer_{horse_no}',
                'finish_position': horse_no  # Simple: horse 1 always wins
            })
    
    return pd.DataFrame(data)


def test_no_future_data_in_rolling_features():
    """Test that rolling features don't use future data."""
    df = create_sample_data()
    
    # Preprocess
    preprocessor = DataPreprocessor()
    df = preprocessor.fit_transform(df)
    
    # Create features
    feature_engineer = FeatureEngineer(n_past_races=3)
    df_with_features = feature_engineer.create_features(df)
    
    # For the first race, past features should be -1 (no data)
    first_race = df_with_features[df_with_features['race_id'] == 'race_0']
    
    assert (first_race['past_1_finish'] == -1).all(), "First race should have no past data"
    assert (first_race['jockey_win_rate'] == -1).all(), "First race should have no jockey stats"
    
    # For later races, check that features only use past data
    # Horse 1 always finishes 1st, so its average should be 1.0
    later_races = df_with_features[df_with_features['race_id'] == 'race_5']
    horse_1_data = later_races[later_races['horse_no'] == 1]
    
    # Past finish should be from previous races only
    assert horse_1_data['past_1_finish'].iloc[0] == 1, "Past finish should be 1"


def test_temporal_ordering():
    """Test that data is properly sorted by date."""
    df = create_sample_data()
    
    # Shuffle data
    df = df.sample(frac=1).reset_index(drop=True)
    
    # Preprocess
    preprocessor = DataPreprocessor()
    df = preprocessor.fit_transform(df)
    
    # Create features
    feature_engineer = FeatureEngineer()
    df_with_features = feature_engineer.create_features(df)
    
    # Check that data is sorted
    assert df_with_features['date'].is_monotonic_increasing, "Data should be sorted by date"


def test_no_leakage_in_jockey_stats():
    """Test that jockey stats don't include current race."""
    df = create_sample_data()
    
    preprocessor = DataPreprocessor()
    df = preprocessor.fit_transform(df)
    
    feature_engineer = FeatureEngineer(n_past_races=3, min_races_for_stats=2)
    df_with_features = feature_engineer.create_features(df)
    
    # For each race, jockey win rate should only use past races
    for race_id in df_with_features['race_id'].unique():
        race_data = df_with_features[df_with_features['race_id'] == race_id]
        race_date = race_data['date'].iloc[0]
        
        for _, row in race_data.iterrows():
            jockey_id = row['jockey_id_encoded']
            jockey_win_rate = row['jockey_win_rate']
            
            if jockey_win_rate != -1:  # Has enough data
                # Calculate expected win rate from past data only
                past_data = df_with_features[
                    (df_with_features['jockey_id_encoded'] == jockey_id) &
                    (df_with_features['date'] < race_date)
                ]
                
                if len(past_data) >= 2:
                    expected_win_rate = (past_data['finish_position'] == 1).mean()
                    # Allow small floating point differences
                    assert abs(jockey_win_rate - expected_win_rate) < 0.01, \
                        f"Jockey win rate should match past data only"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
