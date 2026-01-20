"""
Cross-Sectional Ranking Module

Implements professional universe ranking:
- Converts model probabilities to percentile ranks
- Forces selectivity (top 5% = BUY)
- Reduces noise automatically

This is how real quant funds work.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple


class UniverseRanker:
    """
    Ranks stocks cross-sectionally based on model predictions.
    
    Key Insight:
    Markets don't pay for "will this stock go up?"
    They pay for "which stock will go up MOST?"
    """
    
    def __init__(self):
        self.rank_thresholds = {
            'BUY': 0.95,      # Top 5%
            'WATCH': 0.80,    # Top 20%
            'IGNORE': 0.00    # Bottom 80%
        }
    
    def rank_universe(self, predictions: List[Dict]) -> pd.DataFrame:
        """
        Convert probabilities to percentile ranks.
        
        Args:
            predictions: List of dicts with {'symbol', 'probability', ...}
        
        Returns:
            DataFrame with rank_percentile column
        """
        df = pd.DataFrame(predictions)
        
        if df.empty or 'probability' not in df.columns:
            return df
        
        # Calculate percentile rank (0-1)
        df['rank_percentile'] = df['probability'].rank(pct=True)
        
        # Sort by rank descending
        df = df.sort_values('rank_percentile', ascending=False).reset_index(drop=True)
        
        return df
    
    def classify_by_rank(self, rank_percentile: float) -> str:
        """
        Classify stock based on rank percentile.
        
        Decision thresholds:
        - Top 5%: BUY (best ideas only)
        - Top 20%: WATCH (potential)
        - Bottom 80%: IGNORE (noise)
        
        Returns: 'BUY' | 'WATCH' | 'IGNORE'
        """
        if rank_percentile >= self.rank_thresholds['BUY']:
            return 'BUY'
        elif rank_percentile >= self.rank_thresholds['WATCH']:
            return 'WATCH'
        else:
            return 'IGNORE'
    
    def apply_classification(self, ranked_df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply rank-based classification to entire universe.
        """
        if 'rank_percentile' not in ranked_df.columns:
            return ranked_df
        
        ranked_df['rank_signal'] = ranked_df['rank_percentile'].apply(self.classify_by_rank)
        
        return ranked_df
    
    def calculate_rank_dispersion(self, ranked_df: pd.DataFrame, top_n=5, next_n=15) -> float:
        """
        Calculate rank dispersion (signal clarity).
        
        Logic:
        - Compare top 5 stocks vs next 15
        - If spread is small, signals are noisy
        - If spread is large, top picks are clear
        
        Returns:
            float: Probability spread (0-1)
        """
        if len(ranked_df) < (top_n + next_n):
            return 0.0
        
        top_mean = ranked_df.head(top_n)['probability'].mean()
        next_mean = ranked_df.iloc[top_n:top_n+next_n]['probability'].mean()
        
        spread = top_mean - next_mean
        
        return spread
    
    def should_trade_today(self, ranked_df: pd.DataFrame, min_spread=0.15) -> bool:
        """
        Determine if we should trade based on rank dispersion.
        
        Logic:
        - Only trade when top ranks are clearly separated
        - If spread < threshold, signals are noisy → don't trade
        
        Args:
            ranked_df: Ranked universe
            min_spread: Minimum probability spread required (default 15%)
        
        Returns:
            bool: True if should trade
        """
        spread = self.calculate_rank_dispersion(ranked_df)
        
        return spread >= min_spread
    
    def get_top_picks(self, ranked_df: pd.DataFrame, signal='BUY') -> pd.DataFrame:
        """
        Get stocks matching a specific signal.
        
        Args:
            ranked_df: Ranked universe with classifications
            signal: 'BUY' | 'WATCH' | 'IGNORE'
        
        Returns:
            Filtered DataFrame
        """
        if 'rank_signal' not in ranked_df.columns:
            ranked_df = self.apply_classification(ranked_df)
        
        return ranked_df[ranked_df['rank_signal'] == signal].copy()


def example_usage():
    """Example of how to use the ranking system"""
    
    # Simulate universe predictions
    universe_predictions = [
        {'symbol': 'AAPL', 'probability': 0.75, 'market': 'US'},
        {'symbol': 'NVDA', 'probability': 0.82, 'market': 'US'},
        {'symbol': 'TSLA', 'probability': 0.45, 'market': 'US'},
        {'symbol': 'AMD', 'probability': 0.68, 'market': 'US'},
        {'symbol': 'MSFT', 'probability': 0.79, 'market': 'US'},
        {'symbol': 'GOOGL', 'probability': 0.52, 'market': 'US'},
        {'symbol': 'META', 'probability': 0.71, 'market': 'US'},
        {'symbol': 'AMZN', 'probability': 0.64, 'market': 'US'},
    ]
    
    # Initialize ranker
    ranker = UniverseRanker()
    
    # Rank universe
    ranked = ranker.rank_universe(universe_predictions)
    ranked = ranker.apply_classification(ranked)
    
    print("Universe Ranking:")
    print(ranked[['symbol', 'probability', 'rank_percentile', 'rank_signal']])
    
    # Check dispersion
    spread = ranker.calculate_rank_dispersion(ranked)
    should_trade = ranker.should_trade_today(ranked)
    
    print(f"\nRank Dispersion: {spread:.2%}")
    print(f"Should Trade Today: {should_trade}")
    
    # Get top picks
    buy_signals = ranker.get_top_picks(ranked, signal='BUY')
    print(f"\nBUY Signals ({len(buy_signals)}):")
    print(buy_signals[['symbol', 'probability', 'rank_percentile']])


if __name__ == "__main__":
    example_usage()
