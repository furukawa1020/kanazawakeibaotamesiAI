"""
Scoring pipeline for race prediction.
"""
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np
import logging
from pathlib import Path

from ..models import HorseRacingRanker
from ..data import DataPreprocessor, FeatureEngineer

logger = logging.getLogger(__name__)


class RaceScorer:
    """
    Score horses in a race using trained model.
    """
    
    def __init__(
        self,
        model: HorseRacingRanker,
        preprocessor: DataPreprocessor,
        feature_engineer: FeatureEngineer
    ):
        """
        Initialize scorer.
        
        Args:
            model: Trained ranker model
            preprocessor: Data preprocessor
            feature_engineer: Feature engineer
        """
        self.model = model
        self.preprocessor = preprocessor
        self.feature_engineer = feature_engineer
    
    def score_race(self, race_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """
        Score all horses in a race.
        
        Args:
            race_df: DataFrame with race data (one race, multiple horses)
        
        Returns:
            Tuple of (scores, weights) where weights = exp(scores)
        """
        # Preprocess
        race_df = self.preprocessor.transform(race_df)
        
        # Generate features
        race_df = self.feature_engineer.create_features(race_df)
        
        # Get feature columns
        feature_cols = self.feature_engineer.get_feature_columns(race_df)
        
        # Ensure all required features exist
        missing_features = set(self.model.feature_names) - set(feature_cols)
        if missing_features:
            logger.warning(f"Missing features: {missing_features}. Filling with 0.")
            for feat in missing_features:
                race_df[feat] = 0
        
        # Select features in correct order
        X = race_df[self.model.feature_names]
        
        # Predict scores
        scores = self.model.predict(X)
        
        # Convert to weights
        weights = np.exp(scores)
        
        logger.info(f"Scored {len(scores)} horses. Score range: [{scores.min():.3f}, {scores.max():.3f}]")
        
        return scores, weights
    
    @classmethod
    def load_from_registry(
        cls,
        model_name: str,
        registry_dir: str = "models"
    ) -> 'RaceScorer':
        """
        Load scorer from model registry.
        
        Args:
            model_name: Name of registered model
            registry_dir: Registry directory
        
        Returns:
            Initialized RaceScorer
        """
        from ..models import ModelRegistry
        
        registry = ModelRegistry(registry_dir)
        model_path = registry.get_model_path(model_name)
        
        if model_path is None:
            raise ValueError(f"Model not found in registry: {model_name}")
        
        # Load model
        model = HorseRacingRanker()
        model.load(model_path)
        
        # Load preprocessor and feature engineer
        # These should be saved with the model
        model_dir = Path(model_path).parent
        
        preprocessor_path = model_dir / "preprocessor.pkl"
        feature_engineer_path = model_dir / "feature_engineer.pkl"
        
        import joblib
        
        if preprocessor_path.exists():
            preprocessor = joblib.load(preprocessor_path)
        else:
            logger.warning("Preprocessor not found. Creating new one.")
            preprocessor = DataPreprocessor()
        
        if feature_engineer_path.exists():
            feature_engineer = joblib.load(feature_engineer_path)
        else:
            logger.warning("Feature engineer not found. Creating new one.")
            feature_engineer = FeatureEngineer()
        
        return cls(model, preprocessor, feature_engineer)
