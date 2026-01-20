"""
Yahoo Finance Intraday Data Client

NOTE: This is a TEMPORARY solution for development/backtesting.
For production, use Kite Connect or other professional data vendor.

Fetches 1-minute data and aggregates to 5-minute bars.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional


class YahooIntradayClient:
    """
    Temporary intraday data client using Yahoo Finance.
    
    WARNING: Yahoo Finance intraday data:
    - Limited to last 7 days
    - May have gaps/missing data
    - Not suitable for production trading
    
    Use Kite Connect for production.
    """
    
    def __init__(self):
        self.interval = '1m'  # Fetch 1-minute data
    
    def fetch_intraday_data(self, 
                           symbol: str, 
                           days_back: int = 7) -> pd.DataFrame:
        """
        Fetch 1-minute intraday data.
        
        Args:
            symbol: Stock symbol (e.g., 'RELIANCE.NS')
            days_back: Number of days to fetch (max 7 for Yahoo)
        
        Returns:
            DataFrame with 1-minute OHLCV data
        """
        try:
            # Add .NS suffix if not present
            if not symbol.endswith('.NS') and not symbol.endswith('.BO'):
                symbol = f"{symbol}.NS"
            
            # Fetch data
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=f"{days_back}d", interval=self.interval)
            
            if df.empty:
                print(f"No data for {symbol}")
                return pd.DataFrame()
            
            # Standardize columns
            df = df.reset_index()
            df.columns = [c.lower() for c in df.columns]
            df = df.rename(columns={'datetime': 'timestamp'})
            
            # Add symbol
            df['symbol'] = symbol
            
            return df[['symbol', 'timestamp', 'open', 'high', 'low', 'close', 'volume']]
        
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
            return pd.DataFrame()
    
    def aggregate_to_5min(self, df_1min: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate 1-minute data to 5-minute bars.
        
        Args:
            df_1min: DataFrame with 1-minute data
        
        Returns:
            DataFrame with 5-minute OHLCV data
        """
        if df_1min.empty:
            return pd.DataFrame()
        
        # Set timestamp as index
        df = df_1min.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
        
        # Resample to 5-minute bars
        df_5min = df.resample('5T').agg({
            'symbol': 'first',
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })
        
        # Remove bars with no data
        df_5min = df_5min.dropna(subset=['close'])
        
        # Reset index
        df_5min = df_5min.reset_index()
        
        return df_5min
    
    def filter_trading_hours(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter data to NSE trading hours (09:15 - 15:30 IST).
        
        Args:
            df: DataFrame with timestamp column
        
        Returns:
            Filtered DataFrame
        """
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Extract time
        df['time'] = df['timestamp'].dt.time
        
        # NSE trading hours
        market_open = pd.to_datetime('09:15').time()
        market_close = pd.to_datetime('15:30').time()
        
        # Filter
        df = df[(df['time'] >= market_open) & (df['time'] <= market_close)]
        
        return df.drop(columns=['time'])
    
    def fetch_5min_data(self, symbol: str, days_back: int = 7) -> pd.DataFrame:
        """
        Fetch and aggregate to 5-minute data in one call.
        
        Args:
            symbol: Stock symbol
            days_back: Number of days
        
        Returns:
            5-minute OHLCV DataFrame
        """
        # Fetch 1-min
        df_1min = self.fetch_intraday_data(symbol, days_back)
        
        if df_1min.empty:
            return pd.DataFrame()
        
        # Aggregate to 5-min
        df_5min = self.aggregate_to_5min(df_1min)
        
        # Filter trading hours
        df_5min = self.filter_trading_hours(df_5min)
        
        return df_5min


def example_usage():
    """Example of fetching intraday data"""
    
    client = YahooIntradayClient()
    
    # Fetch 5-min data for RELIANCE
    print("Fetching RELIANCE intraday data...")
    df = client.fetch_5min_data('RELIANCE.NS', days_back=5)
    
    print(f"\nFetched {len(df)} 5-minute bars")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    print("\nSample data:")
    print(df.head(10))
    
    # Stats
    print(f"\nDaily bar counts:")
    df['date'] = df['timestamp'].dt.date
    print(df.groupby('date').size())


if __name__ == "__main__":
    example_usage()
