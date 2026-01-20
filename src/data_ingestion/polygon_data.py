import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional

class PolygonClient:
    BASE_URL = "https://api.polygon.io"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("POLYGON_API_KEY")
        if not self.api_key:
            raise ValueError("Polygon API Key is missing. Set POLYGON_API_KEY env var.")

    def fetch_daily_bars(self, symbol: str, start_date: str, end_date: str = None) -> pd.DataFrame:
        """
        Fetches daily OHLCV bars for a given symbol.
        Range: start_date to end_date (default today).
        Format: YYYY-MM-DD
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        # Endpoint: /v2/aggs/ticker/{stocksTicker}/range/{multiplier}/{timespan}/{from}/{to}
        endpoint = f"/v2/aggs/ticker/{symbol}/range/1/day/{start_date}/{end_date}"
        url = f"{self.BASE_URL}{endpoint}"
        
        params = {
            "adjusted": "true",
            "sort": "asc",
            "limit": 50000,
            "apiKey": self.api_key
        }

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            if data.get("status") != "OK":
                # Some successful queries return status 'OK' even if count is 0.
                if data.get("count", 0) == 0:
                    print(f"Warning: No data found for {symbol} in range.")
                    return pd.DataFrame()
                # If status is ERROR, raise.
                if data.get("status") == "ERROR":
                     raise ValueError(f"Polygon Error: {data.get('error', 'Unknown error')}")

            results = data.get("results", [])
            if not results:
                return pd.DataFrame()

            # Polygon returns: 
            # v: volume, vw: vwap, o: open, c: close, h: high, l: low, t: timestamp (ms), n: transactions
            df = pd.DataFrame(results)
            
            # Convert timestamp to datetime
            df['date'] = pd.to_datetime(df['t'], unit='ms')
            
            # Rename columns
            df = df.rename(columns={
                "o": "open",
                "h": "high",
                "l": "low",
                "c": "close",
                "v": "volume"
            })
            
            # Polygon adjusted=true means 'c' is split adjusted, but not fully dividend adjusted in the aggregates endpoint? 
            # Actually, per docs: "adjusted: whether or not the results are adjusted for splits. By default, results are adjusted."
            # We will treat 'close' as 'adjusted_close' for now or duplicate it.
            df['adjusted_close'] = df['close'] 

            df = df.set_index('date')
            cols = ["open", "high", "low", "close", "adjusted_close", "volume"]
            
            return df[cols]

        except Exception as e:
            print(f"Error fetching Polygon data for {symbol}: {e}")
            raise

if __name__ == "__main__":
    # Test
    # export POLYGON_API_KEY=...
    pass
