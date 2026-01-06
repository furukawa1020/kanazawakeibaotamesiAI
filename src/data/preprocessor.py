"""
Data preprocessing for Kanazawa 3T.
Handles cleaning, encoding, and preparation for feature engineering.
"""
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Preprocess race data for model training."""
    
    def __init__(self):
        """Initialize preprocessor."""
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.fitted = False
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit encoders and transform data.
        
        Args:
            df: Raw race data
        
        Returns:
            Preprocessed dataframe
        """
        df = df.copy()
        
        # Ensure date is datetime
        if 'date' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'])
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        # Encode categorical variables
        categorical_cols = ['surface', 'track_condition', 'class', 'sex', 'jockey_id', 'trainer_id']
        
        for col in categorical_cols:
            if col in df.columns:
                # Create encoder if not exists
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    # Fit on all unique values
                    self.label_encoders[col].fit(df[col].astype(str))
                
                # Transform
                df[f'{col}_encoded'] = self.label_encoders[col].transform(df[col].astype(str))
        
        # Handle horse_id if available
        if 'horse_id' in df.columns:
            if 'horse_id' not in self.label_encoders:
                self.label_encoders['horse_id'] = LabelEncoder()
                self.label_encoders['horse_id'].fit(df['horse_id'].astype(str))
            df['horse_id_encoded'] = self.label_encoders['horse_id'].transform(df['horse_id'].astype(str))
        
        self.fitted = True
        logger.info(f"Fitted preprocessor on {len(df)} rows")
        
        return df
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted encoders.
        
        Args:
            df: Raw race data
        
        Returns:
            Preprocessed dataframe
        """
        if not self.fitted:
            raise ValueError("Preprocessor not fitted. Call fit_transform first.")
        
        df = df.copy()
        
        # Ensure date is datetime
        if 'date' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'])
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        # Encode categorical variables
        for col, encoder in self.label_encoders.items():
            if col in df.columns:
                # Handle unseen categories
                df[f'{col}_encoded'] = df[col].astype(str).apply(
                    lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1
                )
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the data."""
        # For numerical columns, fill with median or -1
        if 'weight_carried' in df.columns:
            df['weight_carried'] = df['weight_carried'].fillna(df['weight_carried'].median())
        
        if 'horse_weight' in df.columns:
            df['horse_weight'] = df['horse_weight'].fillna(df['horse_weight'].median())
        
        if 'horse_weight_diff' in df.columns:
            df['horse_weight_diff'] = df['horse_weight_diff'].fillna(0)
        
        # For categorical columns, fill with 'UNKNOWN'
        categorical_cols = ['surface', 'track_condition', 'class', 'sex', 'jockey_id', 'trainer_id', 'horse_id']
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].fillna('UNKNOWN')
        
        return df
    
    def detect_temporal_leaks(self, df: pd.DataFrame, feature_cols: list) -> Dict[str, Any]:
        """
        Detect potential temporal data leaks.
        
        Args:
            df: Dataframe with features
            feature_cols: List of feature column names
        
        Returns:
            Dictionary with leak detection results
        """
        results = {
            'has_leaks': False,
            'warnings': []
        }
        
        # Check if any features have future information
        # This is a basic check - more sophisticated checks should be added
        
        # Check 1: Ensure date column exists
        if 'date' not in df.columns:
            results['warnings'].append("No 'date' column found - cannot verify temporal ordering")
            return results
        
        # Check 2: Verify data is sorted by date
        if not df['date'].is_monotonic_increasing:
            results['warnings'].append("Data is not sorted by date - potential temporal leak")
            results['has_leaks'] = True
        
        # Check 3: Look for suspicious feature names that might indicate leakage
        leak_keywords = ['future', 'next', 'after', 'result', 'outcome']
        for col in feature_cols:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in leak_keywords):
                results['warnings'].append(f"Suspicious feature name: {col}")
                results['has_leaks'] = True
        
        return results
