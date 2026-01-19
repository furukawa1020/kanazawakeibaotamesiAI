"""
Validation Script for 2025 Data.
Merges historical data with new 2025 data to ensure accurate feature engineering,
then evaluates model performance specifically on Dec 2025.
"""
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import logging

# Add src to path
sys.path.append(str(Path.cwd()))
from src.data.preprocessor import DataPreprocessor
from src.data.features import FeatureEngineer

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def validate_2025():
    # 1. Load History (2020-2024)
    logger.info("Loading historical data (2020-2024)...")
    df_history = pd.read_csv('data/kanazawa_2020_2024_final.csv')
    
    # 2. Load New Data (2025)
    # Check if file exists
    path_2025 = Path('data/kanazawa_2025.csv')
    if not path_2025.exists():
        logger.error(f"2025 data not found at {path_2025}")
        return
        
    logger.info("Loading 2025 data...")
    df_2025 = pd.read_csv(path_2025)
    logger.info(f"Loaded {len(df_2025)} predictions candidates.")
    
    # 3. Merge & Sort
    logger.info("Merging datasets for correct history calculation...")
    df = pd.concat([df_history, df_2025], ignore_index=True)
    
    # Preprocess
    logger.info("Preprocessing...")
    preprocessor = DataPreprocessor()
    df = preprocessor.fit_transform(df)
    
    # Feature Engineering (Vectorized - Fast!)
    logger.info("Generating features...")
    fe = FeatureEngineer()
    df = fe.create_features(df)
    
    # 4. Filter for Verification Target (Dec 2025)
    df['date'] = pd.to_datetime(df['date'])
    # Target: Dec 2025
    target_mask = (df['date'] >= '2025-12-01') & (df['date'] <= '2025-12-31')
    target_df = df[target_mask].copy()
    
    if len(target_df) == 0:
        logger.warning("No data found for Dec 2025! Checking full 2025...")
        target_mask = (df['date'].dt.year == 2025)
        target_df = df[target_mask].copy()
    
    logger.info(f"Validation Target: {len(target_df)} rows")
    
    # 5. Load Model & Predict
    logger.info("Loading trained model...")
    model = lgb.Booster(model_file='models/kanazawa_ranker_v1.txt')
    
    # Select specific features matching training logic
    # Logic from train_final.py lines 75-76
    import numpy as np
    
    # We must apply this to the WHOLE dataframe or ensures target_df has these columns
    # target_df is derived from df which went through FE, so it has them.
    
    numeric_cols = target_df.select_dtypes(include=[np.number]).columns
    feature_cols = [c for c in numeric_cols 
                   if c not in ['finish_position', 'race_id_encoded'] and 
                   ('win_rate' in c or 'encoded' in c or 'weight' in c or 'age' in c or 
                    'distance' in c or 'gate' in c)]
    
    X_target = target_df[feature_cols]
    
    # Predict
    logger.info("Predicting...")
    preds = model.predict(X_target)
    target_df['score'] = preds
    
    # 6. Evaluate Accuracy
    correct = 0
    total = 0
    
    results = []
    
    for race_id, group in target_df.groupby('race_id'):
        # Actual winner
        winner = group[group['finish_position'] == 1]
        if len(winner) == 0: continue
        
        # Predicted winner (highest score)
        pred_winner = group.loc[group['score'].idxmax()]
        
        is_hit = (pred_winner['horse_no'] == winner.iloc[0]['horse_no'])
        if is_hit:
            correct += 1
        total += 1
        
        results.append({
            'date': group['date'].iloc[0],
            'race_id': race_id,
            'race_name': group['race_name'].iloc[0],
            'pred_horse': pred_winner['horse_name'],
            'actual_winner': winner.iloc[0]['horse_name'],
            'hit': 'WIN' if is_hit else 'LOSE'
        })
    
    accuracy = correct / total if total > 0 else 0
    logger.info(f"\n{'='*30}")
    logger.info(f"RESULTS FOR DEC 2025 (or available 2025)")
    logger.info(f"{'='*30}")
    logger.info(f"Top-1 Accuracy: {accuracy:.2%} ({correct}/{total} races)")
    
    # Show recent 5 races
    logger.info("\nRecent 5 Predictions:")
    res_df = pd.DataFrame(results).sort_values('date', ascending=False)
    print(res_df.head(5).to_markdown(index=False))
    
    # Save Report
    report_path = Path('validation_report_dec2025.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"# Model Validation Report (Dec 2025)\n\n")
        f.write(f"## Summary\n")
        f.write(f"- **Top-1 Accuracy**: {accuracy:.2%} ({correct}/{total} races)\n")
        f.write(f"- **Total Races Evaluated**: {total}\n\n")
        f.write(f"## Detailed Results (Last 50 races)\n")
        f.write(res_df.head(50).to_markdown(index=False))
    
    logger.info(f"\n[REPORT SAVED] Full report available at: {report_path.absolute()}")

if __name__ == '__main__':
    validate_2025()
