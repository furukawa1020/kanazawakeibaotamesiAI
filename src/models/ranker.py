"""
LightGBM Ranker implementation for horse racing prediction.
Optimized for GPU acceleration with RTX 5060.
"""
from typing import Optional, Dict, Any, Tuple, List
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
import logging
import joblib
from pathlib import Path

logger = logging.getLogger(__name__)


class HorseRacingRanker:
    """
    LightGBM Ranker for horse racing prediction.
    Uses GPU acceleration for faster training.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize ranker.
        
        Args:
            config: Model configuration dictionary
        """
        self.config = config or self._get_default_config()
        self.model: Optional[lgb.Booster] = None
        self.feature_names: Optional[List[str]] = None
        self.feature_importance_: Optional[pd.DataFrame] = None
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default LightGBM configuration optimized for GPU."""
        return {
            'objective': 'lambdarank',
            'metric': 'ndcg',
            'ndcg_eval_at': [1, 3, 5],
            'num_leaves': 31,
            'learning_rate': 0.05,
            'n_estimators': 500,
            'max_depth': -1,
            'min_child_samples': 20,
            'subsample': 0.8,
            'subsample_freq': 1,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'random_state': 42,
            'verbosity': 1,
            # GPU settings for RTX 5060
            'device': 'gpu',
            'gpu_platform_id': 0,
            'gpu_device_id': 0,
            'gpu_use_dp': False,  # Use single precision for speed
        }
    
    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        group: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        group_val: Optional[pd.Series] = None,
        early_stopping_rounds: int = 50
    ) -> Dict[str, Any]:
        """
        Train the ranker model.
        
        Args:
            X: Training features
            y: Training target (finish_position)
            group: Group sizes for each race
            X_val: Validation features
            y_val: Validation target
            group_val: Validation group sizes
            early_stopping_rounds: Early stopping patience
        
        Returns:
            Training history dictionary
        """
        logger.info(f"Training ranker with {len(X)} samples, {X.shape[1]} features")
        logger.info(f"Using GPU acceleration: {self.config.get('device') == 'gpu'}")
        
        # Store feature names
        self.feature_names = list(X.columns)
        
        # Create datasets
        train_data = lgb.Dataset(
            X,
            label=y,
            group=group,
            feature_name=self.feature_names
        )
        
        valid_sets = [train_data]
        valid_names = ['train']
        
        if X_val is not None and y_val is not None and group_val is not None:
            valid_data = lgb.Dataset(
                X_val,
                label=y_val,
                group=group_val,
                feature_name=self.feature_names,
                reference=train_data
            )
            valid_sets.append(valid_data)
            valid_names.append('valid')
        
        # Training callbacks
        callbacks = [
            lgb.log_evaluation(period=50),
        ]
        
        if len(valid_sets) > 1:
            callbacks.append(lgb.early_stopping(stopping_rounds=early_stopping_rounds))
        
        # Train model
        logger.info("Starting training...")
        self.model = lgb.train(
            self.config,
            train_data,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks
        )
        
        # Calculate feature importance
        self._calculate_feature_importance()
        
        # Get training history
        history = {
            'best_iteration': self.model.best_iteration,
            'best_score': self.model.best_score,
        }
        
        logger.info(f"Training complete. Best iteration: {self.model.best_iteration}")
        
        return history
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict scores for input features.
        
        Args:
            X: Features dataframe
        
        Returns:
            Predicted scores (higher = better ranking)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Ensure features are in correct order
        if self.feature_names is not None:
            X = X[self.feature_names]
        
        scores = self.model.predict(X)
        return scores
    
    def predict_race(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict scores and convert to weights for a single race.
        
        Args:
            X: Features for horses in one race
        
        Returns:
            Tuple of (scores, weights) where weights = exp(scores)
        """
        scores = self.predict(X)
        weights = np.exp(scores)
        return scores, weights
    
    def _calculate_feature_importance(self):
        """Calculate and store feature importance."""
        if self.model is None:
            return
        
        importance_gain = self.model.feature_importance(importance_type='gain')
        importance_split = self.model.feature_importance(importance_type='split')
        
        self.feature_importance_ = pd.DataFrame({
            'feature': self.feature_names,
            'gain': importance_gain,
            'split': importance_split
        }).sort_values('gain', ascending=False)
        
        logger.info(f"Top 10 features by gain:\n{self.feature_importance_.head(10)}")
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance dataframe."""
        if self.feature_importance_ is None:
            raise ValueError("Feature importance not calculated. Train model first.")
        return self.feature_importance_.copy()
    
    def save(self, path: str):
        """
        Save model to disk.
        
        Args:
            path: Path to save model
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save LightGBM model
        model_path = path.with_suffix('.txt')
        self.model.save_model(str(model_path))
        
        # Save metadata
        metadata = {
            'feature_names': self.feature_names,
            'config': self.config,
            'feature_importance': self.feature_importance_
        }
        metadata_path = path.with_suffix('.pkl')
        joblib.dump(metadata, metadata_path)
        
        logger.info(f"Model saved to {model_path}")
    
    def load(self, path: str):
        """
        Load model from disk.
        
        Args:
            path: Path to load model from
        """
        path = Path(path)
        
        # Load LightGBM model
        model_path = path.with_suffix('.txt')
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.model = lgb.Booster(model_file=str(model_path))
        
        # Load metadata
        metadata_path = path.with_suffix('.pkl')
        if metadata_path.exists():
            metadata = joblib.load(metadata_path)
            self.feature_names = metadata.get('feature_names')
            self.config = metadata.get('config')
            self.feature_importance_ = metadata.get('feature_importance')
        
        logger.info(f"Model loaded from {model_path}")


def create_time_series_splits(
    df: pd.DataFrame,
    n_splits: int = 5,
    test_size: float = 0.2
) -> List[Tuple[pd.Index, pd.Index]]:
    """
    Create time-series cross-validation splits.
    
    Args:
        df: Dataframe with 'date' column
        n_splits: Number of splits
        test_size: Proportion of data for test set
    
    Returns:
        List of (train_idx, test_idx) tuples
    """
    # Sort by date
    df = df.sort_values('date').reset_index(drop=True)
    
    # Get unique dates
    unique_dates = df['date'].unique()
    unique_dates = np.sort(unique_dates)
    
    # Calculate split points
    n_dates = len(unique_dates)
    test_dates = int(n_dates * test_size)
    
    splits = []
    
    for i in range(n_splits):
        # Calculate split point
        split_point = int(n_dates * (1 - test_size) * (i + 1) / n_splits)
        
        if split_point + test_dates > n_dates:
            break
        
        train_end_date = unique_dates[split_point]
        test_end_date = unique_dates[min(split_point + test_dates, n_dates - 1)]
        
        # Get indices
        train_idx = df[df['date'] < train_end_date].index
        test_idx = df[(df['date'] >= train_end_date) & (df['date'] <= test_end_date)].index
        
        if len(train_idx) > 0 and len(test_idx) > 0:
            splits.append((train_idx, test_idx))
    
    logger.info(f"Created {len(splits)} time-series splits")
    
    return splits


def prepare_ranking_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, np.ndarray]:
    """
    Prepare data for ranking model.
    
    Args:
        df: Dataframe with features and target
    
    Returns:
        Tuple of (X, y, group) where group is array of race sizes
    """
    # Get group sizes (number of horses per race)
    group = df.groupby('race_id').size().values
    
    # Get features (exclude metadata and target)
    exclude_cols = ['race_id', 'date', 'horse_id', 'horse_no', 'finish_position']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols]
    y = df['finish_position']
    
    return X, y, group
