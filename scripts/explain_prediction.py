"""
Explainable AI (XAI) script for Kanazawa Racing.
Shows WHY the AI predicted a specific result.
"""
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path.cwd()))
from src.data.preprocessor import DataPreprocessor
from src.data.features import FeatureEngineer

def explain_race(race_date_str='2024-12-24'):
    # 1. Load Data & Model
    print("Loading data and model...")
    df = pd.read_csv('data/kanazawa_2020_2024_final.csv')
    model = lgb.Booster(model_file='models/kanazawa_ranker_v1.txt')
    
    # 2. Preprocess (Same as training)
    print("Processing data...")
    preprocessor = DataPreprocessor()
    df = preprocessor.fit_transform(df)
    fe = FeatureEngineer()
    df = fe.create_features(df)
    
    # 3. Select a specific race to explain
    # Filter by date
    target_race = df[df['date'] == race_date_str].iloc[0]['race_id']
    race_df = df[df['race_id'] == target_race].copy()
    
    print(f"\nAnalyzing Race: {target_race} ({race_date_str})")
    print("-" * 50)
    
    # Prepare features
    feature_cols = [c for c in race_df.select_dtypes(include=['number']).columns 
                   if c not in ['finish_position', 'race_id_encoded'] and ('win_rate' in c or 'encoded' in c or 'weight' in c or 'age' in c or 'distance' in c or 'gate' in c)]
    
    X = race_df[feature_cols]
    
    # 4. Predict
    preds = model.predict(X)
    race_df['AI_Score'] = preds
    
    # 5. Explaining the prediction (Feature Contribution)
    # LightGBM can return 'contributions' (SHAP-like values)
    contributions = model.predict(X, pred_contrib=True)
    # The last column is the base value (bias), others correspond to features
    
    # Sort by AI score
    race_df = race_df.sort_values('AI_Score', ascending=False)
    
    print(f"{'Horse':<15} | {'Pred':<5} | {'Actual':<6} | {'Top Reason (Positive)'}")
    print("-" * 70)
    
    for idx, row in race_df.iterrows():
        # Get contribution for this horse (need to map back to X index)
        # Note: 'row' is from sorted df, need its original index in X
        orig_idx = X.index.get_loc(idx)
        contrib = contributions[orig_idx]
        
        # Find biggest positive contributor
        # contrib[:-1] match feature_cols
        best_feat_idx = contrib[:-1].argmax()
        best_feat_name = feature_cols[best_feat_idx]
        best_feat_val = contrib[best_feat_idx]
        
        # Format output
        horse_name = row['horse_name'][:12]
        pred_rank = list(race_df.index).index(idx) + 1
        actual = row['finish_position']
        
        print(f"{horse_name:<15} | {pred_rank:<5} | {actual:<6} | {best_feat_name} (+{best_feat_val:.3f})")

if __name__ == '__main__':
    explain_race('2024-12-24')
