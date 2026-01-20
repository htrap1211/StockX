import pandas as pd
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Any

from src.data_ingestion.polygon_data import PolygonClient
from src.data_ingestion.india_data import IndiaMarketClient
from src.features.engine import FeatureEngine
from src.models.train import ModelTrainer
# Note: In production, we load pre-trained models. 
# For V1, we retrain on the fly or load if available. 
# We'll simulate "Load Latest Model" by retraining quickly on recent history for this MVP loop, 
# OR ideally load from disk. Let's stick to "Train on history, Predict on today" for simplicity in V1.

class RecommendationPipeline:
    def __init__(self):
        self.feature_engine = FeatureEngine()
        # In a real app, we'd load a saved model from 'models/'
        self.trainer = ModelTrainer(model_type="xgboost") 
    
    async def get_recommendation(self, symbol: str, market: str) -> Dict[str, Any]:
        """
        Runs the full pipeline for a single symbol:
        1. Fetch Data (last 300 days to ensure enough for features).
        2. Generate Features.
        3. Train/Load Model.
        4. Predict for *Today*.
        5. Return Recommendation Packet.
        """
        # 1. Fetch Data
        try:
            if market == 'US':
                client = PolygonClient()
                # Fetch 2 years for robust training context
                start_date = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")
                df = client.fetch_daily_bars(symbol, start_date=start_date)
            else:
                client = IndiaMarketClient()
                sym_suffix = symbol if (symbol.endswith('.NS') or symbol.endswith('.BO')) else f"{symbol}.NS"
                df = client.fetch_daily_data(sym_suffix, period="2y")
                
            if df.empty:
                return {"symbol": symbol, "error": "No data found"}
                
        except Exception as e:
            return {"symbol": symbol, "error": str(e)}

        # 2. Features
        df = self.feature_engine.generate_features(df)
        
        # 3. Labeling (for training only)
        # We need to train a model "up to today".
        # In production, this is done offline.
        # Here, we will train on [Start, Today-20] and predict on [Today].
        
        from src.models.labeling import LabelingEngine
        le = LabelingEngine()
        df = le.add_labels(df)
        
        # 4. Train Model
        # Drop rows where target is NaN (recent 20 days)
        valid_train = df.dropna(subset=['Target_Swing'])
        
        if valid_train.empty:
             return {"symbol": symbol, "error": "Not enough data for training"}
             
        feature_cols = [c for c in df.columns if c not in ['open', 'high', 'low', 'close', 'adjusted_close', 'volume', 'Target_Swing', 'Target_LT']]
        X_train = valid_train[feature_cols]
        y_train = valid_train['Target_Swing']
        
        clf = self.trainer.get_model()
        clf.fit(X_train, y_train)
        
        # 5. Predict (Latest available bar)
        last_row = df.iloc[-1:]
        X_last = last_row[feature_cols]
        
        prob = clf.predict_proba(X_last)[:, 1][0]
        
        # 6. Formulate Recommendation
        # Specs: Buy if Prob > 0.6
        side = "BUY" if prob > 0.6 else "WATCH" # We don't really Short in this Swing strategy, or Sell existing? 
        # Requirement: Buy / Watch / Avoid
        if prob > 0.6:
            rec = "BUY"
            risk = "High" if prob < 0.7 else "Medium" # Just a dummy logic, higher prob = lower risk usually? Or higher conviction.
        elif prob < 0.4:
            rec = "AVOID"
            risk = "N/A"
        else:
            rec = "WATCH"
            risk = "Low"

        # Reasoning: Top 3 features contributing? 
        # clf.feature_importances_
        importances = pd.Series(clf.feature_importances_, index=feature_cols).sort_values(ascending=False).head(3)
        reasoning = [f"{idx}: {val:.2f}" for idx, val in importances.items()]

        return {
            "symbol": symbol,
            "market": market,
            "recommendation": rec,
            "confidence_score": f"{prob:.1%}",
            "time_horizon": "Swing (20 days)",
            "entry_price": last_row['close'].values[0],
            "stop_loss": last_row['close'].values[0] * 0.95, # 5% SL
            "reasoning": reasoning,
            "analysis_date": last_row.index[0].strftime("%Y-%m-%d")
        }

if __name__ == "__main__":
    # Test
    async def main():
        pipeline = RecommendationPipeline()
        rec = await pipeline.get_recommendation("NVDA", "US")
        print(rec)
    asyncio.run(main())
