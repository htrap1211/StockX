import asyncio
import pandas as pd
import os
from datetime import datetime, timedelta
from typing import List
from src.data_ingestion.polygon_data import PolygonClient
from src.data_ingestion.india_data import IndiaMarketClient
from src.shared.database import get_db, engine
from src.shared.models import MarketDataDaily, Base

async def save_to_db(df: pd.DataFrame, symbol: str):
    """
    Saves a dataframe of OHLCV data to the database.
    """
    if df.empty:
        return

    async with get_db() as session:
        # This is a naive insert; in production, use upsert/ON CONFLICT or copy
        # For TimescaleDB/Postgres, bulk insert is better.
        # Check if table exists, if not create (should be done by migration/startup, but here for safety)
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        records = []
        for index, row in df.iterrows():
            record = MarketDataDaily(
                symbol=symbol,
                date=index,
                open=row['open'],
                high=row['high'],
                low=row['low'],
                close=row['close'],
                volume=row['volume'],
                adjusted_close=row.get('adjusted_close', None)
            )
            session.add(record)
        
        try:
            await session.commit()
            print(f"Saved {len(df)} records for {symbol}")
        except Exception as e:
            await session.rollback()
            print(f"Failed to save {symbol}: {e}")

async def run_backfill(symbols: List[str], market: str):
    print(f"Starting backfill for {market}: {symbols}")
    
    if market == 'US':
        client = PolygonClient(api_key=os.getenv("POLYGON_API_KEY")) 
        for sym in symbols:
            try:
                print(f"Fetching {sym}...")
                # Fetch last 2 years by default for backfill
                start_date = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")
                df = client.fetch_daily_bars(sym, start_date=start_date)
                await save_to_db(df, sym)
            except Exception as e:
                print(f"Error {sym}: {e}")
                
    elif market == 'IN':
        client = IndiaMarketClient()
        for sym in symbols:
            try:
                print(f"Fetching {sym}...")
                # Ensure suffix
                if not (sym.endswith('.NS') or sym.endswith('.BO')):
                    sym += '.NS' # Default to NSE
                df = client.fetch_daily_data(sym, period="1y") # Backfill 1 year for now
                await save_to_db(df, sym)
            except Exception as e:
                print(f"Error {sym}: {e}")

if __name__ == "__main__":
    # Example usage
    # python src/data_ingestion/backfill.py
    import sys
    
    # Simple CLI args: market symbol1 symbol2 ...
    if len(sys.argv) < 3:
        print("Usage: python backfill.py [US|IN] SYMBOL1 SYMBOL2")
        print("Running demo mode...")
        # asyncio.run(run_backfill(['IBM'], 'US'))
        # asyncio.run(run_backfill(['RELIANCE.NS'], 'IN'))
    else:
        market = sys.argv[1]
        symbols = sys.argv[2:]
        asyncio.run(run_backfill(symbols, market))
