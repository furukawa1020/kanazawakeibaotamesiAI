"""
Data schema definitions and validation for Kanazawa 3T.
"""
from typing import List, Optional, Set
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import numpy as np


class Surface(str, Enum):
    """Track surface types."""
    TURF = "芝"
    DIRT = "ダ"


class TrackCondition(str, Enum):
    """Track condition types."""
    FIRM = "良"
    GOOD = "稍重"
    YIELDING = "重"
    SOFT = "不良"


class RaceClass(str, Enum):
    """Race class types."""
    A = "A"
    B = "B"
    C = "C"


@dataclass
class DataSchema:
    """Schema definition for race data."""
    
    # Required columns
    REQUIRED_COLUMNS: List[str] = None
    
    # Optional columns
    OPTIONAL_COLUMNS: List[str] = None
    
    # Data types
    DTYPES: dict = None
    
    def __post_init__(self):
        """Initialize schema definitions."""
        self.REQUIRED_COLUMNS = [
            'race_id',
            'date',
            'distance',
            'surface',
            'track_condition',
            'class',
            'horse_no',
            'gate',
            'sex',
            'age',
            'weight_carried',
            'jockey_id',
            'trainer_id',
            'finish_position'
        ]
        
        self.OPTIONAL_COLUMNS = [
            'horse_id',
            'time',
            'margin',
            'closing3f',
            'passing_order',
            'horse_weight',
            'horse_weight_diff'
        ]
        
        self.DTYPES = {
            'race_id': 'str',
            'date': 'str',  # Will be converted to datetime
            'distance': 'int',
            'surface': 'str',
            'track_condition': 'str',
            'class': 'str',
            'horse_no': 'int',
            'gate': 'int',
            'sex': 'str',
            'age': 'int',
            'weight_carried': 'float',
            'jockey_id': 'str',
            'trainer_id': 'str',
            'finish_position': 'int',
            'horse_id': 'str',
            'time': 'float',
            'margin': 'float',
            'closing3f': 'float',
            'passing_order': 'str',
            'horse_weight': 'float',
            'horse_weight_diff': 'float'
        }
    
    def validate(self, df: pd.DataFrame) -> tuple[bool, List[str]]:
        """
        Validate dataframe against schema.
        
        Args:
            df: DataFrame to validate
        
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # Check required columns
        missing_cols = set(self.REQUIRED_COLUMNS) - set(df.columns)
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")
        
        # Check data types (for existing columns)
        for col in df.columns:
            if col in self.DTYPES:
                expected_type = self.DTYPES[col]
                
                # Skip type check for columns with all NaN
                if df[col].isna().all():
                    continue
                
                # Check numeric types
                if expected_type in ['int', 'float']:
                    if not pd.api.types.is_numeric_dtype(df[col]):
                        errors.append(f"Column '{col}' should be numeric, got {df[col].dtype}")
        
        # Check value ranges
        if 'finish_position' in df.columns:
            invalid_positions = df[df['finish_position'] < 1]
            if len(invalid_positions) > 0:
                errors.append(f"Invalid finish_position values (< 1): {len(invalid_positions)} rows")
        
        if 'distance' in df.columns:
            invalid_distances = df[(df['distance'] < 800) | (df['distance'] > 3000)]
            if len(invalid_distances) > 0:
                errors.append(f"Suspicious distance values: {len(invalid_distances)} rows")
        
        if 'age' in df.columns:
            invalid_ages = df[(df['age'] < 2) | (df['age'] > 15)]
            if len(invalid_ages) > 0:
                errors.append(f"Suspicious age values: {len(invalid_ages)} rows")
        
        # Check for duplicate race_id + horse_no
        if 'race_id' in df.columns and 'horse_no' in df.columns:
            duplicates = df.duplicated(subset=['race_id', 'horse_no'], keep=False)
            if duplicates.any():
                errors.append(f"Duplicate race_id + horse_no combinations: {duplicates.sum()} rows")
        
        return len(errors) == 0, errors
    
    def get_required_columns(self) -> List[str]:
        """Get list of required columns."""
        return self.REQUIRED_COLUMNS.copy()
    
    def get_all_columns(self) -> List[str]:
        """Get list of all columns (required + optional)."""
        return self.REQUIRED_COLUMNS + self.OPTIONAL_COLUMNS


@dataclass
class OddsSchema:
    """Schema definition for trifecta odds data."""
    
    REQUIRED_COLUMNS: List[str] = None
    
    def __post_init__(self):
        """Initialize odds schema."""
        self.REQUIRED_COLUMNS = [
            'race_id',
            'first',
            'second',
            'third',
            'odds'
        ]
    
    def validate(self, df: pd.DataFrame) -> tuple[bool, List[str]]:
        """
        Validate odds dataframe.
        
        Args:
            df: DataFrame to validate
        
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # Check required columns
        missing_cols = set(self.REQUIRED_COLUMNS) - set(df.columns)
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")
            return False, errors
        
        # Check for invalid combinations (same horse in multiple positions)
        invalid_combos = df[
            (df['first'] == df['second']) |
            (df['first'] == df['third']) |
            (df['second'] == df['third'])
        ]
        if len(invalid_combos) > 0:
            errors.append(f"Invalid trifecta combinations (duplicate horses): {len(invalid_combos)} rows")
        
        # Check odds values
        invalid_odds = df[df['odds'] <= 0]
        if len(invalid_odds) > 0:
            errors.append(f"Invalid odds values (<= 0): {len(invalid_odds)} rows")
        
        return len(errors) == 0, errors
