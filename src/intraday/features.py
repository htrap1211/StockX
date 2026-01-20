"""
Intraday Feature Engine (5-Minute Bars)

Calculates features for intraday setups:
- VWAP
- Distance from VWAP
- EMA 9 / EMA 21
- Opening Range (first 15 min)
- Volume spike ratio
- Intraday volatility
- Time-of-day encoding
"""

import pandas as pd
import numpy as np
from datetime import time


class IntradayFeatureEngine:
    """
    Feature calculator for 5-minute intraday bars.
    """
    
    def __init__(self):
        self.market_open = time(9, 15)
        self.market_close = time(15, 30)
    
    def calculate_vwap(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate Volume Weighted Average Price.
        
        VWAP = Σ(Price × Volume) / Σ(Volume)
        """
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        vwap = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
        
        return vwap
    
    def calculate_distance_from_vwap(self, df: pd.DataFrame) -> pd.Series:
        """
        Distance from VWAP in percentage.
        """
        vwap = self.calculate_vwap(df)
        distance_pct = ((df['close'] - vwap) / vwap) * 100
        
        return distance_pct
    
    def calculate_ema(self, series: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average"""
        return series.ewm(span=period, adjust=False).mean()
    
    def get_opening_range(self, df: pd.DataFrame) -> tuple:
        """
        Get opening range (first 15 minutes).
        
        Returns:
            (or_high, or_low)
        """
        # First 15 minutes = first 3 bars (5-min each)
        opening_bars = df.head(3)
        
        if len(opening_bars) < 3:
            return None, None
        
        or_high = opening_bars['high'].max()
        or_low = opening_bars['low'].min()
        
        return or_high, or_low
    
    def calculate_volume_spike_ratio(self, df: pd.DataFrame, lookback: int = 20) -> pd.Series:
        """
        Volume spike ratio: current / rolling average.
        """
        rolling_avg_volume = df['volume'].rolling(lookback).mean()
        spike_ratio = df['volume'] / rolling_avg_volume
        
        return spike_ratio
    
    def calculate_intraday_volatility(self, df: pd.DataFrame, lookback: int = 20) -> pd.Series:
        """
        Intraday volatility (rolling std of returns).
        """
        returns = df['close'].pct_change()
        volatility = returns.rolling(lookback).std()
        
        return volatility
    
    def encode_time_of_day(self, df: pd.DataFrame) -> pd.Series:
        """
        Minutes since market open.
        """
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Calculate minutes since 09:15
        market_open_today = df['timestamp'].dt.normalize() + pd.Timedelta(hours=9, minutes=15)
        minutes_since_open = (df['timestamp'] - market_open_today).dt.total_seconds() / 60
        
        return minutes_since_open
    
    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all intraday features.
        
        Args:
            df: 5-minute OHLCV DataFrame
        
        Returns:
            DataFrame with features added
        """
        df = df.copy()
        
        # VWAP features
        df['vwap'] = self.calculate_vwap(df)
        df['distance_from_vwap_pct'] = self.calculate_distance_from_vwap(df)
        
        # EMA
        df['ema_9'] = self.calculate_ema(df['close'], 9)
        df['ema_21'] = self.calculate_ema(df['close'], 21)
        
        # Opening range
        or_high, or_low = self.get_opening_range(df)
        df['opening_range_high'] = or_high
        df['opening_range_low'] = or_low
        
        # Volume
        df['volume_spike_ratio'] = self.calculate_volume_spike_ratio(df)
        
        # Volatility
        df['intraday_volatility'] = self.calculate_intraday_volatility(df)
        
        # Time encoding
        df['minutes_since_open'] = self.encode_time_of_day(df)
        
        return df


def example_usage():
    """Example of feature generation"""
    
    # Load sample data
    from src.intraday.data_ingestion.yahoo_intraday import YahooIntradayClient
    
    client = YahooIntradayClient()
    df = client.fetch_5min_data('RELIANCE.NS', days_back=2)
    
    if df.empty:
        print("No data fetched")
        return
    
    # Generate features
    feature_engine = IntradayFeatureEngine()
    df_features = feature_engine.generate_features(df)
    
    print(f"Generated features for {len(df_features)} bars\n")
    
    # Show sample
    print("Sample features:")
    print(df_features[['timestamp', 'close', 'vwap', 'distance_from_vwap_pct', 
                       'ema_9', 'volume_spike_ratio']].tail(10))
    
    # Opening range
    print(f"\nOpening Range:")
    print(f"  High: {df_features['opening_range_high'].iloc[0]:.2f}")
    print(f"  Low: {df_features['opening_range_low'].iloc[0]:.2f}")


if __name__ == "__main__":
    example_usage()
