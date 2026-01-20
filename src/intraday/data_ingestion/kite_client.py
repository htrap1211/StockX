"""
Kite Connect Client for NSE Intraday Data

Professional-grade data source for India intraday trading.
Replaces Yahoo Finance for production use.

Setup:
1. Create account at https://kite.zerodha.com/
2. Get API key from https://developers.kite.trade/
3. Set environment variables:
   - KITE_API_KEY
   - KITE_API_SECRET
   - KITE_ACCESS_TOKEN (generated via login flow)
"""

import os
from datetime import datetime, timedelta
import pandas as pd
from kiteconnect import KiteConnect
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KiteIntradayClient:
    """
    Kite Connect client for NSE intraday data.
    
    Advantages over Yahoo Finance:
    - No gaps in data
    - Sub-second latency
    - Reliable 1-minute bars
    - Official NSE data
    - WebSocket support for real-time
    """
    
    def __init__(self, api_key=None, access_token=None):
        """
        Initialize Kite Connect client.
        
        Args:
            api_key: Kite API key (or from env KITE_API_KEY)
            access_token: Access token (or from env KITE_ACCESS_TOKEN)
        """
        self.api_key = api_key or os.getenv("KITE_API_KEY")
        self.access_token = access_token or os.getenv("KITE_ACCESS_TOKEN")
        
        if not self.api_key:
            raise ValueError("Kite API Key missing. Set KITE_API_KEY env var.")
        
        if not self.access_token:
            logger.warning("Access token missing. You'll need to complete login flow.")
            logger.info("Run: python src/intraday/data_ingestion/kite_login.py")
        
        self.kite = KiteConnect(api_key=self.api_key)
        
        if self.access_token:
            self.kite.set_access_token(self.access_token)
    
    def fetch_historical_data(self, 
                             symbol: str,
                             from_date: datetime,
                             to_date: datetime,
                             interval: str = "5minute") -> pd.DataFrame:
        """
        Fetch historical intraday data.
        
        Args:
            symbol: NSE symbol (e.g., 'RELIANCE')
            from_date: Start date
            to_date: End date
            interval: '1minute', '5minute', '15minute', etc.
        
        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Kite uses instrument tokens, not symbols
            # For demo, we'll use symbol directly
            # In production, map symbol -> instrument_token
            
            instrument_token = self._get_instrument_token(symbol)
            
            data = self.kite.historical_data(
                instrument_token=instrument_token,
                from_date=from_date,
                to_date=to_date,
                interval=interval
            )
            
            if not data:
                logger.warning(f"No data for {symbol}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            df['symbol'] = symbol
            
            # Standardize columns
            df = df.rename(columns={
                'date': 'timestamp',
                'volume': 'volume'
            })
            
            return df[['symbol', 'timestamp', 'open', 'high', 'low', 'close', 'volume']]
        
        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
            return pd.DataFrame()
    
    def _get_instrument_token(self, symbol: str) -> int:
        """
        Get instrument token for symbol.
        
        In production, maintain a mapping file or query instruments API.
        For now, return placeholder.
        """
        # Placeholder - in production, use:
        # instruments = self.kite.instruments("NSE")
        # token = [i for i in instruments if i['tradingsymbol'] == symbol][0]['instrument_token']
        
        # Common NSE tokens (hardcoded for demo)
        token_map = {
            'RELIANCE': 738561,
            'TCS': 2953217,
            'INFY': 408065,
            'HDFCBANK': 341249,
            'ICICIBANK': 1270529
        }
        
        return token_map.get(symbol, 0)
    
    def fetch_5min_data(self, symbol: str, days_back: int = 7) -> pd.DataFrame:
        """
        Fetch 5-minute data for recent days.
        
        Args:
            symbol: NSE symbol
            days_back: Number of days to fetch
        
        Returns:
            5-minute OHLCV DataFrame
        """
        to_date = datetime.now()
        from_date = to_date - timedelta(days=days_back)
        
        return self.fetch_historical_data(
            symbol=symbol,
            from_date=from_date,
            to_date=to_date,
            interval="5minute"
        )


def setup_instructions():
    """Print setup instructions for Kite Connect"""
    print("=" * 60)
    print("KITE CONNECT SETUP INSTRUCTIONS")
    print("=" * 60)
    print()
    print("1. Create Kite account:")
    print("   https://kite.zerodha.com/")
    print()
    print("2. Get API credentials:")
    print("   https://developers.kite.trade/")
    print("   - Create new app")
    print("   - Get API Key and API Secret")
    print()
    print("3. Set environment variables:")
    print("   export KITE_API_KEY='your_api_key'")
    print("   export KITE_API_SECRET='your_api_secret'")
    print()
    print("4. Generate access token:")
    print("   python src/intraday/data_ingestion/kite_login.py")
    print()
    print("5. Cost: ₹2,000/month")
    print()
    print("=" * 60)
    print()
    print("NOTE: For now, we'll continue using Yahoo Finance.")
    print("      Kite Connect requires manual setup and paid subscription.")
    print("=" * 60)


if __name__ == "__main__":
    setup_instructions()
