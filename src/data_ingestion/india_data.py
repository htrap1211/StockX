import yfinance as yf
import pandas as pd
from typing import Optional

class IndiaMarketClient:
    """
    Client for fetching India market data (NSE/BSE).
    Currently uses yfinance as a fallback until official vendor credentials are provided.
    """
    
    def fetch_daily_data(self, symbol: str, period: str = "max") -> pd.DataFrame:
        """
        Fetches daily OHLCV data for an Indian stock.
        Symbol should include suffix (e.g., 'RELIANCE.NS' for NSE, 'TCS.BO' for BSE).
        """
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period)
            
            if df.empty:
                print(f"Warning: No data found for {symbol}")
                return pd.DataFrame()

            # yfinance returns: Open, High, Low, Close, Volume, Dividends, Stock Splits
            # Rename columns to standard schema
            df = df.rename(columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume"
            })
            
            # Create adjusted_close if not present (yfinance 'Close' is adjusted for splits usually, but let's check)
            # Actually yfinance history 'Close' is adjusted for splits? 
            # yfinance has auto_adjust=True by default in some versions.
            # Let's rely on 'Close' as the primary price. 
            
            df["adjusted_close"] = df["close"] # Placeholder if we don't distinguish
            
            df.index.name = "date"
            
            cols = ["open", "high", "low", "close", "adjusted_close", "volume"]
            return df[cols]

        except Exception as e:
            print(f"Error fetching India data for {symbol}: {e}")
            raise

if __name__ == "__main__":
    client = IndiaMarketClient()
    df = client.fetch_daily_data("RELIANCE.NS", period="1mo")
    print(df.head())
