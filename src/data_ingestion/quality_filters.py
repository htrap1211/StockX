"""
Data Quality Filters Module

Filters universe to ensure:
1. Sufficient history (>= 300 trading days)
2. Adequate liquidity (volume thresholds)
3. Data completeness (< 5% missing)

This is non-negotiable for sound backtesting.
"""

import pandas as pd
import numpy as np
from typing import Dict, List
from datetime import datetime, timedelta


class DataQualityFilter:
    """
    Filters stocks based on data quality criteria.
    """
    
    def __init__(self, 
                 min_trading_days=300,
                 min_avg_volume_us=100000,
                 min_avg_volume_in=50000,
                 max_missing_pct=0.05):
        """
        Args:
            min_trading_days: Minimum trading days required
            min_avg_volume_us: Min avg daily volume for US stocks
            min_avg_volume_in: Min avg daily volume for India stocks
            max_missing_pct: Max % of missing data allowed
        """
        self.min_trading_days = min_trading_days
        self.min_avg_volume_us = min_avg_volume_us
        self.min_avg_volume_in = min_avg_volume_in
        self.max_missing_pct = max_missing_pct
    
    def filter_by_history(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter stocks with insufficient trading history.
        
        Args:
            df: DataFrame with 'symbol' column
        
        Returns:
            Filtered DataFrame
        """
        counts = df.groupby('symbol').size()
        valid_symbols = counts[counts >= self.min_trading_days].index
        
        filtered = df[df['symbol'].isin(valid_symbols)]
        
        dropped = len(counts) - len(valid_symbols)
        print(f"  Dropped {dropped} stocks with < {self.min_trading_days} days")
        
        return filtered
    
    def filter_by_volume(self, df: pd.DataFrame, market: str) -> pd.DataFrame:
        """
        Filter low-liquidity stocks.
        
        Args:
            df: DataFrame with 'symbol', 'volume' columns
            market: 'US' or 'IN'
        
        Returns:
            Filtered DataFrame
        """
        threshold = self.min_avg_volume_us if market == 'US' else self.min_avg_volume_in
        
        avg_volume = df.groupby('symbol')['volume'].mean()
        liquid_symbols = avg_volume[avg_volume >= threshold].index
        
        filtered = df[df['symbol'].isin(liquid_symbols)]
        
        dropped = len(avg_volume) - len(liquid_symbols)
        print(f"  Dropped {dropped} stocks with avg volume < {threshold:,.0f}")
        
        return filtered
    
    def filter_by_missing_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter stocks with excessive missing data.
        
        Args:
            df: DataFrame with price/volume data
        
        Returns:
            Filtered DataFrame
        """
        # Calculate missing % per symbol
        missing_pct = df.groupby('symbol').apply(
            lambda x: x[['close', 'volume']].isnull().sum().sum() / (len(x) * 2)
        )
        
        clean_symbols = missing_pct[missing_pct <= self.max_missing_pct].index
        
        filtered = df[df['symbol'].isin(clean_symbols)]
        
        dropped = len(missing_pct) - len(clean_symbols)
        print(f"  Dropped {dropped} stocks with > {self.max_missing_pct:.0%} missing data")
        
        return filtered
    
    def apply_all_filters(self, df: pd.DataFrame, market: str) -> pd.DataFrame:
        """
        Apply all quality filters sequentially.
        
        Args:
            df: Raw market data
            market: 'US' or 'IN'
        
        Returns:
            Clean, tradable universe
        """
        print(f"\n{market} Market - Quality Filters:")
        print(f"  Starting: {df['symbol'].nunique()} stocks")
        
        # Filter 1: History
        df = self.filter_by_history(df)
        print(f"  After history filter: {df['symbol'].nunique()} stocks")
        
        # Filter 2: Volume
        df = self.filter_by_volume(df, market)
        print(f"  After volume filter: {df['symbol'].nunique()} stocks")
        
        # Filter 3: Missing data
        df = self.filter_by_missing_data(df)
        print(f"  Final clean universe: {df['symbol'].nunique()} stocks")
        
        return df
    
    def get_quality_report(self, df: pd.DataFrame) -> Dict:
        """
        Generate quality report for universe.
        
        Returns:
            Dict with quality metrics
        """
        report = {}
        
        # Trading days per symbol
        days_per_symbol = df.groupby('symbol').size()
        report['avg_trading_days'] = days_per_symbol.mean()
        report['min_trading_days'] = days_per_symbol.min()
        report['max_trading_days'] = days_per_symbol.max()
        
        # Volume stats
        avg_volume = df.groupby('symbol')['volume'].mean()
        report['avg_volume_mean'] = avg_volume.mean()
        report['avg_volume_median'] = avg_volume.median()
        
        # Missing data
        missing_pct = df.groupby('symbol').apply(
            lambda x: x[['close', 'volume']].isnull().sum().sum() / (len(x) * 2)
        )
        report['avg_missing_pct'] = missing_pct.mean()
        report['max_missing_pct'] = missing_pct.max()
        
        return report


def example_usage():
    """Example of applying quality filters"""
    
    # Simulate market data
    dates = pd.date_range('2022-01-01', '2024-01-01', freq='B')
    
    # Good stock
    good_stock = pd.DataFrame({
        'symbol': 'AAPL',
        'date': dates,
        'close': np.random.randn(len(dates)) + 150,
        'volume': np.random.randint(50000000, 100000000, len(dates))
    })
    
    # Bad stock (low volume)
    bad_stock = pd.DataFrame({
        'symbol': 'LOWVOL',
        'date': dates[:100],  # Insufficient history
        'close': np.random.randn(100) + 10,
        'volume': np.random.randint(1000, 5000, 100)  # Low volume
    })
    
    df = pd.concat([good_stock, bad_stock], ignore_index=True)
    
    # Apply filters
    filter_engine = DataQualityFilter()
    clean_df = filter_engine.apply_all_filters(df, market='US')
    
    print(f"\nFinal symbols: {clean_df['symbol'].unique()}")
    
    # Quality report
    report = filter_engine.get_quality_report(clean_df)
    print(f"\nQuality Report:")
    for key, val in report.items():
        print(f"  {key}: {val:.2f}")


if __name__ == "__main__":
    example_usage()
