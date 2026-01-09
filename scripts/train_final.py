"""
Final Model Training Script using collected Kanazawa data (2020-2024).
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import logging
from sklearn.metrics import ndcg_score

# Add src to path
sys.path.append(str(Path.cwd()))

from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.data.features import FeatureEngineer

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def train_model():
    # 1. Load Data
    data_path = Path('data/kanazawa_2020_2024_final.csv')
    logger.info(f"Loading data from {data_path}...")
    
    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} records. Date range: {df['date'].min()} to {df['date'].max()}")
    
    # 2. Preprocess
    logger.info("Preprocessing...")
    preprocessor = DataPreprocessor()
    df = preprocessor.fit_transform(df)
    
    # 3. Feature Engineering
    logger.info("Generating features (this may take a moment)...")
    fe = FeatureEngineer()
    df = fe.create_features(df)
    
    # Drop rows where critical features might be NaN (start of history)
    df = df.dropna(subset=['jockey_win_rate', 'horse_win_rate'], how='all')
    
    # 4. Train/Test Split (Time-based)
    # Train: 2020-2023
    # Test: 2024
    df['date'] = pd.to_datetime(df['date'])
    train_mask = df['date'].dt.year <= 2023
    test_mask = df['date'].dt.year == 2024
    
    train_df = df[train_mask].copy()
    test_df = df[test_mask].copy()
    
    logger.info(f"Train set (2020-2023): {len(train_df)} rows")
    logger.info(f"Test set (2024): {len(test_df)} rows")
    
    # Prepare for LightGBM Ranker
    # We need to sort by race_id to group queries
    train_df = train_df.sort_values('race_id')
    test_df = test_df.sort_values('race_id')
    
    # Define groups (number of horses per race)
    q_train = train_df.groupby('race_id').size().to_numpy()
    q_test = test_df.groupby('race_id').size().to_numpy()
    
    # Features & Targets
    # Exclude non-feature columns
    exclude_cols = ['race_id', 'date', 'race_name', 'horse_name', 'jockey_name', 'trainer_name', 'finish_position', 'time', 'margin']
    feature_cols = [c for c in train_df.columns if c not in exclude_cols and not c.endswith('_encoded') == False] # Include encoded, numeric
    # Actually, let's select numeric features specifically or use the feature list from FE
    
    # Simple selection for now: all numeric columns except target and ID
    feature_cols = [c for c in train_df.select_dtypes(include=[np.number]).columns 
                   if c not in ['finish_position', 'race_id_encoded'] and 'win_rate' in c or 'encoded' in c or 'weight' in c or 'age' in c or 'distance' in c or 'gate' in c]
    
    logger.info(f"Training with {len(feature_cols)} features: {feature_cols[:5]}...")
    
    X_train = train_df[feature_cols]
    y_train = train_df['finish_position']
    X_test = test_df[feature_cols]
    y_test = test_df['finish_position']
    
    # For LambdaRank, target should be relevance (higher is better). 
    # Finish position (1st is best) needs inverting.
    # Relevance = 1 / finish_position (or just arbitrary scores like 1st=10, 2nd=5, etc)
    # Simple inversion: 18 - finish_position (assuming max 18 horses)
    y_train_rel = (18 - y_train).clip(0, 18)
    y_test_rel = (18 - y_test).clip(0, 18)
    
    # 5. Train
    logger.info("Starting LightGBM Training on GPU...")
    
    model = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        # GPU Parameters - FIXED for RTX 5060
        device="gpu", 
        gpu_platform_id=1, # 1 = NVIDIA RTX 5060, 0 = Intel
        gpu_device_id=0,
        max_position=18
    )
    
    try:
        model.fit(
            X_train, y_train_rel,
            group=q_train,
            eval_set=[(X_test, y_test_rel)],
            eval_group=[q_test],
            eval_at=[1, 3, 5],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)]
        )
    except Exception as e:
        logger.error(f"GPU Training failed: {e}")
        logger.info("Retrying with CPU...")
        model.set_params(device='cpu')
        model.fit(
            X_train, y_train_rel,
            group=q_train,
            eval_set=[(X_test, y_test_rel)],
            eval_group=[q_test],
            eval_at=[1, 3, 5],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)]
        )
    
    # 6. Evaluate
    logger.info("Evaluating...")
    
    # specific metric: Top-1 Accuracy
    # Predict scores
    test_pred = model.predict(X_test)
    test_df['score'] = test_pred
    
    # Check if predicted top-1 was actual winner
    correct = 0
    total = 0
    
    for race_id, group in test_df.groupby('race_id'):
        # Actual winner
        winner = group[group['finish_position'] == 1]
        if len(winner) == 0: continue
        
        # Predicted winner (highest score)
        pred_winner = group.loc[group['score'].idxmax()]
        
        if pred_winner['horse_no'] == winner.iloc[0]['horse_no']:
            correct += 1
        total += 1
    
    accuracy = correct / total if total > 0 else 0
    logger.info(f"Top-1 Accuracy on 2024 Test Data: {accuracy:.2%} ({correct}/{total} races)")
    
    # Feature Importance
    logger.info("\nFeature Importance:")
    imp_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(imp_df.head(10).to_markdown())
    
    # Save model
    model_path = Path('models/kanazawa_ranker_v1.txt')
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(model_path)
    logger.info(f"Model saved to {model_path}")

if __name__ == '__main__':
    train_model()
