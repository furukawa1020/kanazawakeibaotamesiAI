"""
Data loading and validation for Kanazawa 3T.
"""
from pathlib import Path
from typing import Optional, Union
import pandas as pd
import logging

from .schema import DataSchema, OddsSchema

logger = logging.getLogger(__name__)


class DataLoader:
    """Load and validate race data."""
    
    def __init__(self, schema: Optional[DataSchema] = None):
        """
        Initialize data loader.
        
        Args:
            schema: Data schema for validation. If None, uses default DataSchema.
        """
        self.schema = schema or DataSchema()
    
    def load_races(
        self,
        file_path: Union[str, Path],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        validate: bool = True
    ) -> pd.DataFrame:
        """
        Load race data from file.
        
        Args:
            file_path: Path to data file (CSV or Parquet)
            start_date: Optional start date filter (YYYY-MM-DD)
            end_date: Optional end date filter (YYYY-MM-DD)
            validate: Whether to validate against schema
        
        Returns:
            DataFrame with race data
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        # Load based on file extension
        if file_path.suffix == '.csv':
            df = pd.read_csv(file_path)
        elif file_path.suffix == '.parquet':
            df = pd.read_parquet(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        logger.info(f"Loaded {len(df)} rows from {file_path}")
        
        # Convert date column to datetime
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        
        # Filter by date range
        if start_date:
            start_date = pd.to_datetime(start_date)
            df = df[df['date'] >= start_date]
            logger.info(f"Filtered to {len(df)} rows after start_date={start_date}")
        
        if end_date:
            end_date = pd.to_datetime(end_date)
            df = df[df['date'] <= end_date]
            logger.info(f"Filtered to {len(df)} rows after end_date={end_date}")
        
        # Validate
        if validate:
            is_valid, errors = self.schema.validate(df)
            if not is_valid:
                error_msg = "Data validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
                raise ValueError(error_msg)
            logger.info("Data validation passed")
        
        # Sort by date and race_id for consistency
        df = df.sort_values(['date', 'race_id', 'horse_no']).reset_index(drop=True)
        
        return df
    
    def load_odds(
        self,
        file_path: Union[str, Path],
        race_ids: Optional[list] = None,
        validate: bool = True
    ) -> pd.DataFrame:
        """
        Load trifecta odds data.
        
        Args:
            file_path: Path to odds file
            race_ids: Optional list of race IDs to filter
            validate: Whether to validate against schema
        
        Returns:
            DataFrame with odds data
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Odds file not found: {file_path}")
        
        # Load based on file extension
        if file_path.suffix == '.csv':
            df = pd.read_csv(file_path)
        elif file_path.suffix == '.parquet':
            df = pd.read_parquet(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        logger.info(f"Loaded {len(df)} odds combinations from {file_path}")
        
        # Filter by race IDs
        if race_ids is not None:
            df = df[df['race_id'].isin(race_ids)]
            logger.info(f"Filtered to {len(df)} odds for {len(race_ids)} races")
        
        # Validate
        if validate:
            odds_schema = OddsSchema()
            is_valid, errors = odds_schema.validate(df)
            if not is_valid:
                error_msg = "Odds validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
                raise ValueError(error_msg)
            logger.info("Odds validation passed")
        
        return df
    
    def save_data(
        self,
        df: pd.DataFrame,
        file_path: Union[str, Path],
        format: str = 'parquet'
    ):
        """
        Save dataframe to file.
        
        Args:
            df: DataFrame to save
            file_path: Output file path
            format: Output format ('csv' or 'parquet')
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'csv':
            df.to_csv(file_path, index=False)
        elif format == 'parquet':
            df.to_parquet(file_path, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Saved {len(df)} rows to {file_path}")
