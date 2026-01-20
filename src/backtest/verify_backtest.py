import asyncio
import pandas as pd
from src.shared.database import get_db
from src.shared.models import MarketDataDaily
from sqlalchemy import select
from src.features.engine import FeatureEngine
from src.models.labeling import LabelingEngine
from src.models.train import ModelTrainer
from src.backtest.simulation import BacktestEngine

async def run_backtest_verification():
    print("1. Loading data (AAPL)...")
    async with get_db() as session:
        result = await session.execute(
             select(MarketDataDaily).where(MarketDataDaily.symbol == 'AAPL').order_by(MarketDataDaily.date)
        )
        rows = result.scalars().all()
        data = [{'date': r.date, 'open': r.open, 'high': r.high, 'low': r.low, 
                 'close': r.close, 'volume': r.volume, 'adjusted_close': r.adjusted_close} for r in rows]
        df = pd.DataFrame(data).set_index('date')
    
    # 2. Pipeline
    print("2. Generating Features & Labels...")
    feat_engine = FeatureEngine()
    df = feat_engine.generate_features(df)
    
    label_engine = LabelingEngine()
    df = label_engine.add_labels(df)
    
    # 3. Train Model (to get predictions)
    print("3. Training Model to generate predictions...")
    feature_cols = [c for c in df.columns if c not in ['open', 'high', 'low', 'close', 
                                                       'adjusted_close', 'volume', 'Target_Swing', 'Target_LT']]
    trainer = ModelTrainer(model_type="xgboost")
    
    # We use the same train_walk_forward to get out-of-sample predictions
    metrics, preds_df = trainer.train_walk_forward(df, target_col='Target_Swing', feature_cols=feature_cols)
    print(f"   Model Metrics: ROC-AUC={metrics['roc_auc']:.2f}")
    
    # preds_df has ['date', 'actual', 'pred', 'prob']
    preds_df = preds_df.set_index('date')
    
    # 4. Run Backtest
    print("4. Running Backtest...")
    bt_engine = BacktestEngine()
    results = bt_engine.run_simulation(df, preds_df) # Pass original df for prices
    
    bt_engine.generate_report(results)

if __name__ == "__main__":
    asyncio.run(run_backtest_verification())
