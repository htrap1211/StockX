import asyncio
import pandas as pd
from src.shared.database import get_db
from src.shared.models import MarketDataDaily
from sqlalchemy import select
from src.features.engine import FeatureEngine
from src.models.labeling import LabelingEngine
from src.models.train import ModelTrainer

async def run_training_verification():
    print("Loading data...")
    async with get_db() as session:
        # Fetch AAPL
        result = await session.execute(
             select(MarketDataDaily).where(MarketDataDaily.symbol == 'AAPL').order_by(MarketDataDaily.date)
        )
        rows = result.scalars().all()
        data = [{'date': r.date, 'open': r.open, 'high': r.high, 'low': r.low, 'close': r.close, 'volume': r.volume, 'adjusted_close': r.adjusted_close} for r in rows]
        df = pd.DataFrame(data).set_index('date')
    
    print(f"Data loaded: {len(df)} rows")
    
    # 1. Feature Engineering
    print("Generating Features...")
    feat_engine = FeatureEngine()
    df = feat_engine.generate_features(df)
    
    # 2. Labeling
    print("Generating Labels...")
    label_engine = LabelingEngine()
    df = label_engine.add_labels(df)
    
    # Check class balance
    print(f"Swing Targets (1s): {df['Target_Swing'].sum()} / {len(df)}")
    
    # 3. Training
    print("Starting Walk-Forward Training (XGBoost)...")
    feature_cols = [c for c in df.columns if c not in ['open', 'high', 'low', 'close', 'adjusted_close', 'volume', 'Target_Swing', 'Target_LT']]
    
    trainer = ModelTrainer(model_type="xgboost")
    metrics, res_df = trainer.train_walk_forward(df, target_col='Target_Swing', feature_cols=feature_cols)
    
    print("Training Complete.")
    print("Metrics:", metrics)
    print("Sample Predictions:")
    print(res_df.tail())

if __name__ == "__main__":
    asyncio.run(run_training_verification())
