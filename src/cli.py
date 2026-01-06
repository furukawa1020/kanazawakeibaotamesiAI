"""
Command-line interface for Kanazawa 3T.
"""
import click
import logging
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from utils import load_config, setup_logger
from data import DataLoader, DataPreprocessor, FeatureEngineer
from models import HorseRacingRanker, prepare_ranking_data, create_time_series_splits, ModelRegistry
from inference import RaceScorer, PlackettLuceSampler
from betting import TrifectaGenerator, BetFilter, StakeAllocator
from evaluation import Backtester

logger = logging.getLogger(__name__)


@click.group()
@click.option('--config', default='configs/default.yaml', help='Path to config file')
@click.option('--verbose', is_flag=True, help='Enable verbose logging')
@click.pass_context
def cli(ctx, config, verbose):
    """Kanazawa 3T - 競馬予想AI"""
    # Load config
    ctx.ensure_object(dict)
    ctx.obj['config'] = load_config(config)
    
    # Setup logging
    log_level = 'DEBUG' if verbose else ctx.obj['config'].get('logging.level', 'INFO')
    log_file = ctx.obj['config'].get('logging.file', 'logs/keibaai.log')
    
    global logger
    logger = setup_logger('keibaai', level=log_level, log_file=log_file)
    
    logger.info("=" * 60)
    logger.info("Kanazawa 3T - 競馬予想AI")
    logger.info("=" * 60)


@cli.command()
@click.option('--data-path', required=True, help='Path to race data file')
@click.option('--start-date', required=True, help='Start date (YYYY-MM-DD)')
@click.option('--end-date', required=True, help='End date (YYYY-MM-DD)')
@click.option('--model-name', default='ranker_v1', help='Model name for registry')
@click.option('--output-dir', default='models', help='Output directory')
@click.pass_context
def train(ctx, data_path, start_date, end_date, model_name, output_dir):
    """Train ranking model on historical data."""
    config = ctx.obj['config']
    
    logger.info(f"Training model: {model_name}")
    logger.info(f"Data: {data_path}")
    logger.info(f"Period: {start_date} to {end_date}")
    
    # Load data
    logger.info("Loading data...")
    loader = DataLoader()
    df = loader.load_races(data_path, start_date=start_date, end_date=end_date)
    
    # Preprocess
    logger.info("Preprocessing...")
    preprocessor = DataPreprocessor()
    df = preprocessor.fit_transform(df)
    
    # Feature engineering
    logger.info("Creating features...")
    feature_engineer = FeatureEngineer(
        n_past_races=config.get('data.features.n_past_races', 3),
        min_races_for_stats=config.get('data.features.min_races_for_stats', 2)
    )
    df = feature_engineer.create_features(df)
    
    # Prepare ranking data
    logger.info("Preparing ranking data...")
    X, y, group = prepare_ranking_data(df)
    
    # Time-series split
    logger.info("Creating train/validation split...")
    splits = create_time_series_splits(df, n_splits=1, test_size=0.2)
    train_idx, val_idx = splits[0]
    
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    
    # Get group sizes
    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]
    group_train = train_df.groupby('race_id').size().values
    group_val = val_df.groupby('race_id').size().values
    
    # Train model
    logger.info("Training LightGBM Ranker...")
    model_config = config.get('model.params', {})
    ranker = HorseRacingRanker(config=model_config)
    
    history = ranker.train(
        X_train, y_train, group_train,
        X_val, y_val, group_val,
        early_stopping_rounds=config.get('model.early_stopping_rounds', 50)
    )
    
    # Save model
    output_path = Path(output_dir) / model_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving model to {output_path}...")
    ranker.save(str(output_path))
    
    # Save preprocessor and feature engineer
    import joblib
    joblib.dump(preprocessor, output_path.parent / "preprocessor.pkl")
    joblib.dump(feature_engineer, output_path.parent / "feature_engineer.pkl")
    
    # Register model
    registry = ModelRegistry(output_dir)
    registry.register_model(
        model_name,
        str(output_path),
        metadata={
            'start_date': start_date,
            'end_date': end_date,
            'n_features': len(X.columns),
            'best_iteration': history['best_iteration'],
            'best_score': history['best_score']
        }
    )
    
    logger.info("✓ Training complete!")
    logger.info(f"Model saved: {output_path}")


@cli.command()
@click.option('--model-name', required=True, help='Model name from registry')
@click.option('--race-file', required=True, help='Path to race data file')
@click.option('--odds-file', required=True, help='Path to trifecta odds file')
@click.option('--output', required=True, help='Output file path (JSON or CSV)')
@click.option('--n-samples', default=50000, help='Number of Monte Carlo samples')
@click.pass_context
def predict(ctx, model_name, race_file, odds_file, output, n_samples):
    """Generate trifecta bets for a race."""
    config = ctx.obj['config']
    
    logger.info(f"Generating predictions using model: {model_name}")
    
    # Load model
    logger.info("Loading model...")
    scorer = RaceScorer.load_from_registry(model_name)
    
    # Load race data
    logger.info(f"Loading race data from {race_file}...")
    loader = DataLoader()
    race_df = loader.load_races(race_file, validate=False)
    
    # Load odds
    logger.info(f"Loading odds from {odds_file}...")
    odds_df = loader.load_odds(odds_file)
    
    # Score race
    logger.info("Scoring horses...")
    scores, weights = scorer.score_race(race_df)
    
    # Sample probabilities
    logger.info(f"Sampling probabilities (n={n_samples})...")
    sampler = PlackettLuceSampler(
        n_samples=n_samples,
        use_gpu=config.get('inference.probability.use_gpu', True),
        random_seed=config.get('random_seed', 42)
    )
    
    top_k = config.get('betting.top_k_horses', 7)
    trifecta_probs = sampler.estimate_trifecta_probabilities(weights, top_k=top_k)
    
    # Generate candidates
    logger.info("Generating bet candidates...")
    generator = TrifectaGenerator(
        top_k_horses=top_k,
        method=config.get('betting.generation_method', 'sampling')
    )
    candidates = generator.generate_candidates(weights, trifecta_probs)
    
    # Filter by EV
    logger.info("Filtering by expected value...")
    bet_filter = BetFilter(
        min_ev=config.get('betting.filter.min_ev', 1.15),
        min_probability=config.get('betting.filter.min_probability', 0.002),
        max_bets=config.get('betting.filter.max_bets', 30)
    )
    
    # Create odds dictionary
    odds_dict = {
        (row['first'], row['second'], row['third']): row['odds']
        for _, row in odds_df.iterrows()
    }
    
    filtered_bets = bet_filter.filter_candidates(candidates, trifecta_probs, odds_dict)
    
    # Allocate stakes
    logger.info("Allocating stakes...")
    allocator = StakeAllocator(
        budget=config.get('betting.allocation.budget', 3000),
        stake_unit=config.get('betting.allocation.stake_unit', 100),
        min_stake=config.get('betting.allocation.min_stake', 100),
        probability_exponent=config.get('betting.allocation.probability_exponent', 0.7)
    )
    
    final_bets = allocator.allocate(filtered_bets)
    
    # Format output
    output_df = allocator.format_bets_for_output(final_bets)
    
    # Calculate expected return
    expected_return = allocator.calculate_expected_return(final_bets)
    
    # Save output
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if output_path.suffix == '.json':
        import json
        result = {
            'bets': output_df.to_dict('records'),
            'summary': expected_return
        }
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
    else:
        output_df.to_csv(output_path, index=False)
    
    logger.info(f"✓ Generated {len(final_bets)} bets")
    logger.info(f"Total stake: {expected_return['total_stake']} yen")
    logger.info(f"Expected ROI: {expected_return['roi']:.2f}%")
    logger.info(f"Output saved to: {output_path}")


@cli.command()
@click.option('--model-name', required=True, help='Model name from registry')
@click.option('--data-path', required=True, help='Path to race data file')
@click.option('--odds-path', required=True, help='Path to odds data file')
@click.option('--start-date', required=True, help='Start date (YYYY-MM-DD)')
@click.option('--end-date', required=True, help='End date (YYYY-MM-DD)')
@click.option('--output-dir', default='output/backtest', help='Output directory')
@click.option('--compare-baselines', is_flag=True, help='Compare against baseline strategies')
@click.pass_context
def backtest(ctx, model_name, data_path, odds_path, start_date, end_date, output_dir, compare_baselines):
    """Run backtest on historical data."""
    config = ctx.obj['config']
    
    logger.info(f"Running backtest: {start_date} to {end_date}")
    
    # Load data
    logger.info("Loading data...")
    loader = DataLoader()
    races_df = loader.load_races(data_path, start_date=start_date, end_date=end_date)
    odds_df = loader.load_odds(odds_path)
    
    # Results are in races_df (finish_position column)
    results_df = races_df[['race_id', 'horse_no', 'finish_position']].copy()
    
    # Load model
    logger.info(f"Loading model: {model_name}")
    scorer = RaceScorer.load_from_registry(model_name)
    
    # Create bet generator function
    def model_bet_generator(race_data, race_odds):
        # Score
        scores, weights = scorer.score_race(race_data)
        
        # Sample
        sampler = PlackettLuceSampler(
            n_samples=config.get('inference.probability.n_samples', 50000),
            use_gpu=config.get('inference.probability.use_gpu', True),
            random_seed=config.get('random_seed', 42)
        )
        top_k = config.get('betting.top_k_horses', 7)
        trifecta_probs = sampler.estimate_trifecta_probabilities(weights, top_k=top_k)
        
        # Generate candidates
        generator = TrifectaGenerator(top_k_horses=top_k)
        candidates = generator.generate_candidates(weights, trifecta_probs)
        
        # Filter
        bet_filter = BetFilter(
            min_ev=config.get('betting.filter.min_ev', 1.15),
            min_probability=config.get('betting.filter.min_probability', 0.002),
            max_bets=config.get('betting.filter.max_bets', 30)
        )
        
        odds_dict = {
            (row['first'], row['second'], row['third']): row['odds']
            for _, row in race_odds.iterrows()
        }
        
        filtered_bets = bet_filter.filter_candidates(candidates, trifecta_probs, odds_dict)
        
        # Allocate
        allocator = StakeAllocator(
            budget=config.get('betting.allocation.budget', 3000)
        )
        
        return allocator.allocate(filtered_bets)
    
    # Run backtest
    backtester = Backtester(start_date, end_date, output_dir)
    
    if compare_baselines:
        logger.info("Comparing against baseline strategies...")
        comparison = backtester.compare_strategies(
            races_df, odds_df, results_df,
            model_bet_generator,
            baseline_names=config.get('baselines', ['popularity', 'last_race', 'jockey'])
        )
        
        logger.info("\n" + "=" * 60)
        logger.info("STRATEGY COMPARISON")
        logger.info("=" * 60)
        logger.info("\n" + comparison.to_string(index=False))
    else:
        results = backtester.run(
            races_df, odds_df, results_df,
            model_bet_generator,
            strategy_name=model_name
        )
        
        logger.info("\n" + "=" * 60)
        logger.info("BACKTEST RESULTS")
        logger.info("=" * 60)
        metrics = results['metrics']
        for key, value in metrics.items():
            logger.info(f"{key}: {value}")
    
    logger.info(f"\n✓ Backtest complete! Results saved to: {output_dir}")


if __name__ == '__main__':
    cli(obj={})
