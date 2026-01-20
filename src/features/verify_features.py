import asyncio
import pandas as pd
from src.shared.database import get_db
from src.shared.models import MarketDataDaily
from src.features.engine import FeatureEngine
from sqlalchemy import select

async def verify_features():
    async with get_db() as session:
        # Fetch AAPL data
        result = await session.execute(
            select(MarketDataDaily).where(MarketDataDaily.symbol == 'AAPL').order_by(MarketDataDaily.date)
        )
        rows = result.scalars().all()
        
        if not rows:
            print("No data found for AAPL to verify features.")
            return

        # Convert to DataFrame
        data = [
            {
             'date': r.date, 
             'open': r.open, 
             'high': r.high, 
             'low': r.low, 
             'close': r.close, 
             'volume': r.volume, 
             'adjusted_close': r.adjusted_close
            } 
            for r in rows
        ]
        df = pd.DataFrame(data).set_index('date')
        
        print(f"Loaded {len(df)} rows for AAPL.")
        
        engine = FeatureEngine()
        df_rich = engine.generate_features(df)
        
        print("Features generated:")
        print(df_rich.columns.tolist())
        print("\nLast 5 rows:")
        print(df_rich[['close', 'RSI_14', 'MACD_12_26', 'BB_percent_b', 'vol_z_score']].tail())
        
        # Check for look-ahead? 
        # Feature Engine calculates on rolling windows of PAST data, so it is safe by definition 
        # as long as we use shift(1) for target generation (next step).
        
        # Validate some values
        if df_rich['RSI_14'].max() > 100 or df_rich['RSI_14'].min() < 0:
            print("ERROR: RSI out of bounds!")
        else:
            print("RSI bounds check passed.")

if __name__ == "__main__":
    asyncio.run(verify_features())
