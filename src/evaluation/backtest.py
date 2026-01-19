"""
Backtesting framework for evaluating betting strategies.
"""
from typing import Dict, List, Optional, Callable
import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
from tqdm import tqdm

from .metrics import MetricsCalculator
from .baselines import get_baseline_strategy

logger = logging.getLogger(__name__)


class Backtester:
    """
    Backtest betting strategies on historical data.
    """
    
    def __init__(
        self,
        start_date: str,
        end_date: str,
        output_dir: str = "output/backtest"
    ):
        """
        Initialize backtester.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            output_dir: Directory for output files
        """
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics_calculator = MetricsCalculator()
    
    def run(
        self,
        races_df: pd.DataFrame,
        odds_df: pd.DataFrame,
        results_df: pd.DataFrame,
        bet_generator: Callable,
        strategy_name: str = "model"
    ) -> Dict:
        """
        Run backtest.
        
        Args:
            races_df: DataFrame with race data
            odds_df: DataFrame with trifecta odds
            results_df: DataFrame with actual results (finish positions)
            bet_generator: Function that generates bets for a race
            strategy_name: Name of the strategy
        
        Returns:
            Dictionary with backtest results
        """
        logger.info(f"Running backtest: {self.start_date} to {self.end_date}")
        
        # Filter data by date range
        races_df = races_df[
            (races_df['date'] >= self.start_date) &
            (races_df['date'] <= self.end_date)
        ].copy()
        
        # Get unique races
        race_ids = races_df['race_id'].unique()
        logger.info(f"Backtesting on {len(race_ids)} races")
        
        # Track all bets
        all_bets = []
        
        # Process each race
        for race_id in tqdm(race_ids, desc="Backtesting"):
            # Get race data
            race_data = races_df[races_df['race_id'] == race_id]
            race_odds = odds_df[odds_df['race_id'] == race_id]
            race_results = results_df[results_df['race_id'] == race_id]
            
            if len(race_odds) == 0:
                logger.debug(f"No odds for race {race_id}, skipping")
                continue
            
            # Generate bets
            try:
                bets_df = bet_generator(race_data, race_odds)
            except Exception as e:
                logger.warning(f"Error generating bets for race {race_id}: {e}")
                continue
            
            if len(bets_df) == 0:
                continue
            
            # Calculate payouts
            bets_df = self._calculate_payouts(bets_df, race_results)
            bets_df['race_id'] = race_id
            bets_df['date'] = race_data['date'].iloc[0]
            
            all_bets.append(bets_df)
        
        # Combine all bets
        if len(all_bets) == 0:
            logger.warning("No bets generated in backtest period")
            return self._empty_results()
        
        all_bets_df = pd.concat(all_bets, ignore_index=True)
        
        # Calculate metrics
        metrics = self.metrics_calculator.calculate_betting_metrics(all_bets_df)
        
        # Calculate time series metrics
        time_series = self._calculate_time_series(all_bets_df)
        
        # Save results
        results = {
            'strategy': strategy_name,
            'start_date': str(self.start_date.date()),
            'end_date': str(self.end_date.date()),
            'metrics': metrics,
            'time_series': time_series
        }
        
        self._save_results(results, all_bets_df, strategy_name)
        
        logger.info(f"Backtest complete. ROI: {metrics['roi']:.2f}%")
        
        return results
    
    def _calculate_payouts(
        self,
        bets_df: pd.DataFrame,
        results_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate payouts for bets based on actual results.
        
        Args:
            bets_df: DataFrame with bets
            results_df: DataFrame with actual finish positions
        
        Returns:
            DataFrame with 'payout' column added
        """
        # Get actual top 3
        top3 = results_df.nsmallest(3, 'finish_position')['horse_no'].tolist()
        
        if len(top3) < 3:
            # Race didn't complete properly
            bets_df['payout'] = 0
            return bets_df
        
        actual_trifecta = tuple(top3)
        
        # Check each bet
        payouts = []
        for _, bet in bets_df.iterrows():
            bet_trifecta = (bet['first'], bet['second'], bet['third'])
            
            if bet_trifecta == actual_trifecta:
                # Win! Payout = stake * odds / 100
                payout = bet['stake'] * bet['odds'] / 100
            else:
                # Loss
                payout = 0
            
            payouts.append(payout)
        
        bets_df['payout'] = payouts
        
        return bets_df
    
    def _calculate_time_series(self, bets_df: pd.DataFrame) -> Dict:
        """Calculate time series of cumulative P&L."""
        # Group by date
        daily = bets_df.groupby('date').agg({
            'stake': 'sum',
            'payout': 'sum'
        })
        
        daily['pnl'] = daily['payout'] - daily['stake']
        daily['cumulative_pnl'] = daily['pnl'].cumsum()
        
        return {
            'dates': [str(d.date()) for d in daily.index],
            'cumulative_pnl': daily['cumulative_pnl'].tolist(),
            'daily_pnl': daily['pnl'].tolist()
        }
    
    def _save_results(
        self,
        results: Dict,
        bets_df: pd.DataFrame,
        strategy_name: str
    ):
        """Save backtest results to files."""
        # Save summary JSON
        summary_path = self.output_dir / f"{strategy_name}_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Save detailed bets CSV
        bets_path = self.output_dir / f"{strategy_name}_bets.csv"
        bets_df.to_csv(bets_path, index=False)
        
        logger.info(f"Results saved to {self.output_dir}")
    
    def _empty_results(self) -> Dict:
        """Return empty results structure."""
        return {
            'strategy': 'unknown',
            'start_date': str(self.start_date.date()),
            'end_date': str(self.end_date.date()),
            'metrics': {
                'total_races': 0,
                'total_bets': 0,
                'roi': 0
            },
            'time_series': {
                'dates': [],
                'cumulative_pnl': [],
                'daily_pnl': []
            }
        }
    
    def compare_strategies(
        self,
        races_df: pd.DataFrame,
        odds_df: pd.DataFrame,
        results_df: pd.DataFrame,
        model_generator: Callable,
        baseline_names: List[str] = ['popularity', 'last_race', 'jockey']
    ) -> pd.DataFrame:
        """
        Compare model strategy against baselines.
        
        Args:
            races_df: Race data
            odds_df: Odds data
            results_df: Results data
            model_generator: Model bet generator function
            baseline_names: List of baseline strategy names
        
        Returns:
            DataFrame comparing all strategies
        """
        logger.info("Comparing strategies...")
        
        all_results = []
        
        # Run model strategy
        model_results = self.run(
            races_df, odds_df, results_df,
            model_generator, "model"
        )
        all_results.append(model_results)
        
        # Run baseline strategies
        for baseline_name in baseline_names:
            baseline_strategy = get_baseline_strategy(baseline_name)
            
            def baseline_generator(race_data, race_odds):
                return baseline_strategy.generate_bets(race_data, race_odds)
            
            baseline_results = self.run(
                races_df, odds_df, results_df,
                baseline_generator, baseline_name
            )
            all_results.append(baseline_results)
        
        # Create comparison table
        comparison = []
        for result in all_results:
            metrics = result['metrics']
            comparison.append({
                'strategy': result['strategy'],
                'total_races': metrics.get('total_races', 0),
                'total_bets': metrics.get('total_bets', 0),
                'total_stake': metrics.get('total_stake', 0),
                'total_payout': metrics.get('total_payout', 0),
                'roi': metrics.get('roi', 0),
                'hit_rate': metrics.get('hit_rate', 0),
                'max_drawdown': metrics.get('max_drawdown', 0)
            })
        
        comparison_df = pd.DataFrame(comparison)
        
        # Save comparison
        comparison_path = self.output_dir / "strategy_comparison.csv"
        comparison_df.to_csv(comparison_path, index=False)
        
        logger.info(f"Strategy comparison saved to {comparison_path}")
        
        return comparison_df
