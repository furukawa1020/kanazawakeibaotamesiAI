"""
Generate sample race data for testing.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import argparse
from pathlib import Path


def generate_sample_races(n_races=100, horses_per_race=12, start_date='2023-01-01'):
    """Generate sample race data."""
    np.random.seed(42)
    
    start = datetime.strptime(start_date, '%Y-%m-%d')
    
    data = []
    
    for race_idx in range(n_races):
        race_id = f'kanazawa_{start.year}{start.month:02d}{start.day:02d}_R{(race_idx % 12) + 1}'
        race_date = start + timedelta(days=race_idx * 3)  # Race every 3 days
        
        # Race conditions
        distance = np.random.choice([1400, 1500, 1600, 1700, 1900, 2000])
        surface = 'ダ'  # Kanazawa is mostly dirt
        track_condition = np.random.choice(['良', '稍重', '重', '不良'], p=[0.6, 0.2, 0.15, 0.05])
        race_class = np.random.choice(['A', 'B', 'C'], p=[0.2, 0.5, 0.3])
        
        # Generate horses
        n_horses = horses_per_race
        
        # Simulate horse strengths
        strengths = np.random.exponential(1.0, n_horses)
        finish_order = np.argsort(-strengths)  # Higher strength = better finish
        
        for horse_no in range(1, n_horses + 1):
            finish_position = np.where(finish_order == (horse_no - 1))[0][0] + 1
            
            data.append({
                'race_id': race_id,
                'date': race_date.strftime('%Y-%m-%d'),
                'distance': distance,
                'surface': surface,
                'track_condition': track_condition,
                'class': race_class,
                'horse_no': horse_no,
                'horse_id': f'horse_{np.random.randint(1, 500)}',  # Reuse horses
                'gate': horse_no,
                'sex': np.random.choice(['M', 'F', 'G'], p=[0.5, 0.3, 0.2]),
                'age': np.random.randint(3, 8),
                'weight_carried': np.random.uniform(52, 58),
                'jockey_id': f'jockey_{np.random.randint(1, 50)}',
                'trainer_id': f'trainer_{np.random.randint(1, 30)}',
                'finish_position': finish_position,
                'horse_weight': np.random.uniform(440, 520),
                'horse_weight_diff': np.random.uniform(-10, 10)
            })
    
    return pd.DataFrame(data)


def generate_sample_odds(races_df, coverage=0.3):
    """Generate sample trifecta odds."""
    np.random.seed(42)
    
    odds_data = []
    
    for race_id in races_df['race_id'].unique():
        race_horses = races_df[races_df['race_id'] == race_id]['horse_no'].tolist()
        n_horses = len(race_horses)
        
        # Generate odds for a subset of combinations
        n_combinations = int(n_horses * (n_horses - 1) * (n_horses - 2) * coverage)
        
        for _ in range(n_combinations):
            # Random trifecta
            combo = np.random.choice(race_horses, size=3, replace=False)
            
            # Generate odds (log-normal distribution)
            odds = np.random.lognormal(mean=6, sigma=1.5)
            odds = max(100, min(100000, odds))  # Clamp to reasonable range
            
            odds_data.append({
                'race_id': race_id,
                'first': int(combo[0]),
                'second': int(combo[1]),
                'third': int(combo[2]),
                'odds': round(odds, 1)
            })
    
    return pd.DataFrame(odds_data)


def main():
    parser = argparse.ArgumentParser(description='Generate sample race data')
    parser.add_argument('--n-races', type=int, default=100, help='Number of races')
    parser.add_argument('--horses-per-race', type=int, default=12, help='Horses per race')
    parser.add_argument('--output-dir', default='data', help='Output directory')
    parser.add_argument('--start-date', default='2023-01-01', help='Start date')
    
    args = parser.parse_args()
    
    print(f"Generating {args.n_races} sample races...")
    
    # Generate races
    races_df = generate_sample_races(
        n_races=args.n_races,
        horses_per_race=args.horses_per_race,
        start_date=args.start_date
    )
    
    # Generate odds
    print("Generating trifecta odds...")
    odds_df = generate_sample_odds(races_df, coverage=0.3)
    
    # Save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    races_path = output_dir / 'sample_races.csv'
    odds_path = output_dir / 'sample_odds.csv'
    
    races_df.to_csv(races_path, index=False)
    odds_df.to_csv(odds_path, index=False)
    
    print(f"✓ Saved {len(races_df)} race records to {races_path}")
    print(f"✓ Saved {len(odds_df)} odds records to {odds_path}")
    print(f"\nSample statistics:")
    print(f"  - Races: {races_df['race_id'].nunique()}")
    print(f"  - Date range: {races_df['date'].min()} to {races_df['date'].max()}")
    print(f"  - Unique horses: {races_df['horse_id'].nunique()}")
    print(f"  - Unique jockeys: {races_df['jockey_id'].nunique()}")


if __name__ == '__main__':
    main()
