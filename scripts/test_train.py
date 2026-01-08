"""
Script to check if data exists and run a quick training test.
"""
import pandas as pd
from pathlib import Path
import sys
import logging

# Add src to path
sys.path.append(str(Path.cwd()))

from src.models.ranker import Ranker
from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.data.features import FeatureEngineer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def quick_train_test():
    """Run a quick training test with whatever data is available."""
    data_dir = Path('data')
    csv_files = list(data_dir.glob('kanazawa_*.csv'))
    
    if not csv_files:
        logger.warning("No data files found yet.")
        return
    
    logger.info(f"Found {len(csv_files)} data files.")
    
    # Load most recent file
    latest_file = sorted(csv_files)[-1] 
    logger.info(f"Loading {latest_file}...")
    
    try:
        loader = DataLoader()
        # Skip validation for now to be lenient with partial data
        df = pd.read_csv(latest_file)
        logger.info(f"Loaded {len(df)} rows.")
        
        # Basic preprocessing
        preprocessor = DataPreprocessor()
        df = preprocessor.fit_transform(df)
        
        # Feature Engineering (simplified)
        fe = FeatureEngineer()
        df = fe.create_features(df)
        
        # Split train/test (simple split for test)
        train_df = df.iloc[:int(len(df)*0.8)]
        valid_df = df.iloc[int(len(df)*0.8):]
        
        # Train
        logger.info("Starting quick training...")
        ranker = Ranker(params={'n_estimators': 10}) # Very small for speed
        ranker.fit(train_df, valid_df)
        
        logger.info("Training complete!")
        
    except Exception as e:
        logger.error(f"Training test failed: {e}")

if __name__ == '__main__':
    quick_train_test()
