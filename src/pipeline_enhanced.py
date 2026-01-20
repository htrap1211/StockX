"""
Enhanced Recommendation Pipeline with Quant Enhancements

Integrates:
1. Market regime detection
2. Cross-sectional ranking
3. Regime-aware decision logic
"""

import pandas as pd
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Any

from src.data_ingestion.polygon_data import PolygonClient
from src.data_ingestion.india_data import IndiaMarketClient
from src.features.engine import FeatureEngine
from src.features.regimes import RegimeDetector
from src.models.train import ModelTrainer
from src.models.ranking import UniverseRanker


class EnhancedRecommendationPipeline:
    """
    Professional pipeline with regime awareness and cross-sectional ranking.
    """
    
    def __init__(self):
        self.feature_engine = FeatureEngine()
        self.trainer = ModelTrainer(model_type="xgboost")
        self.ranker = UniverseRanker()
    
    async def get_universe_recommendations(self, 
                                          symbols: List[str], 
                                          market: str) -> List[Dict[str, Any]]:
        """
        Get recommendations for entire universe with ranking.
        
        This is the professional way:
        1. Get regime
        2. Run inference on all symbols
        3. Rank cross-sectionally
        4. Apply rank-based classification
        
        Returns:
            List of recommendations with rank info
        """
        # 1. Get current regime
        regime_detector = RegimeDetector(market=market)
        current_regime = regime_detector.get_current_regime()
        
        # 2. Run inference on all symbols
        tasks = [self._get_single_prediction(sym, market, current_regime) for sym in symbols]
        predictions = await asyncio.gather(*tasks)
        
        # Filter out errors
        valid_predictions = [p for p in predictions if 'error' not in p]
        
        if not valid_predictions:
            return []
        
        # 3. Rank cross-sectionally
        ranked_df = self.ranker.rank_universe(valid_predictions)
        ranked_df = self.ranker.apply_classification(ranked_df)
        
        # 4. Check if we should trade today
        should_trade = self.ranker.should_trade_today(ranked_df)
        rank_dispersion = self.ranker.calculate_rank_dispersion(ranked_df)
        
        # 5. Convert back to list of dicts with rank info
        recommendations = []
        for _, row in ranked_df.iterrows():
            rec = row.to_dict()
            rec['rank_percentile'] = f"{row['rank_percentile']:.1%}"
            rec['rank_signal'] = row['rank_signal']
            rec['regime'] = current_regime
            rec['should_trade_today'] = should_trade
            rec['rank_dispersion'] = f"{rank_dispersion:.1%}"
            recommendations.append(rec)
        
        return recommendations
    
    async def _get_single_prediction(self, 
                                    symbol: str, 
                                    market: str,
                                    regime: Dict) -> Dict[str, Any]:
        """
        Get prediction for single symbol with regime features.
        """
        try:
            # Fetch data
            if market == 'US':
                client = PolygonClient()
                start_date = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")
                df = client.fetch_daily_bars(symbol, start_date=start_date)
            else:
                client = IndiaMarketClient()
                sym_suffix = symbol if (symbol.endswith('.NS') or symbol.endswith('.BO')) else f"{symbol}.NS"
                df = client.fetch_daily_data(sym_suffix, period="2y")
            
            if df.empty:
                return {"symbol": symbol, "error": "No data found"}
            
            # Generate features
            df = self.feature_engine.generate_features(df)
            
            # Add regime features
            regime_detector = RegimeDetector(market=market)
            regime_features = regime_detector.encode_regime_features(regime)
            for key, val in regime_features.items():
                df[key] = val
            
            # Add labels for training
            from src.models.labeling import LabelingEngine
            le = LabelingEngine()
            df = le.add_labels(df)
            
            # Train model
            valid_train = df.dropna(subset=['Target_Swing'])
            if valid_train.empty:
                return {"symbol": symbol, "error": "Not enough data"}
            
            feature_cols = [c for c in df.columns if c not in 
                          ['open', 'high', 'low', 'close', 'adjusted_close', 'volume', 
                           'Target_Swing', 'Target_LT']]
            
            X_train = valid_train[feature_cols]
            y_train = valid_train['Target_Swing']
            
            clf = self.trainer.get_model()
            clf.fit(X_train, y_train)
            
            # Predict on latest
            last_row = df.iloc[-1:]
            X_last = last_row[feature_cols]
            
            probability = float(clf.predict_proba(X_last)[:, 1][0])
            
            # Calculate volatility for portfolio sizing
            returns = df['close'].pct_change()
            volatility_20d = float(returns.rolling(20).std().iloc[-1] * (252 ** 0.5))
            
            # Get feature importances
            importances = pd.Series(clf.feature_importances_, index=feature_cols).sort_values(ascending=False).head(3)
            reasoning = [f"{idx}: {val:.2f}" for idx, val in importances.items()]
            
            return {
                "symbol": symbol,
                "market": market,
                "probability": probability,
                "confidence_score": f"{probability:.1%}",
                "time_horizon": "Swing (20 days)",
                "entry_price": float(last_row['close'].values[0]),
                "stop_loss": float(last_row['close'].values[0] * 0.95),
                "volatility_20d": volatility_20d,
                "reasoning": reasoning,
                "analysis_date": last_row.index[0].strftime("%Y-%m-%d"),
                "sector": "UNKNOWN"  # Would need sector mapping
            }
            
        except Exception as e:
            return {"symbol": symbol, "error": str(e)}


# Backward compatibility: keep old pipeline for single-stock queries
class RecommendationPipeline:
    """Original pipeline for backward compatibility"""
    
    def __init__(self):
        self.enhanced = EnhancedRecommendationPipeline()
    
    async def get_recommendation(self, symbol: str, market: str) -> Dict[str, Any]:
        """Get single stock recommendation (legacy method)"""
        results = await self.enhanced.get_universe_recommendations([symbol], market)
        
        if not results:
            return {"symbol": symbol, "error": "No recommendation generated"}
        
        rec = results[0]
        
        # Convert to legacy format
        if 'rank_signal' in rec:
            recommendation = rec['rank_signal']
        else:
            prob = rec.get('probability', 0)
            recommendation = "BUY" if prob > 0.6 else "AVOID" if prob < 0.4 else "WATCH"
        
        return {
            "symbol": rec['symbol'],
            "market": rec['market'],
            "recommendation": recommendation,
            "confidence_score": rec['confidence_score'],
            "time_horizon": rec['time_horizon'],
            "entry_price": rec['entry_price'],
            "stop_loss": rec['stop_loss'],
            "reasoning": rec['reasoning'],
            "analysis_date": rec['analysis_date']
        }


if __name__ == "__main__":
    async def test():
        pipeline = EnhancedRecommendationPipeline()
        
        # Test universe ranking
        symbols = ["AAPL", "NVDA", "TSLA", "AMD", "MSFT"]
        recs = await pipeline.get_universe_recommendations(symbols, "US")
        
        print(f"Generated {len(recs)} recommendations")
        for rec in recs[:3]:
            print(f"\n{rec['symbol']}: {rec.get('rank_signal', 'N/A')} "
                  f"(Rank: {rec.get('rank_percentile', 'N/A')}, "
                  f"Prob: {rec.get('confidence_score', 'N/A')})")
    
    asyncio.run(test())
