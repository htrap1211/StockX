"""
Intraday Inference Pipeline

Generates real-time intraday signals using trained ML models.
"""

import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime


class IntradayInference:
    """
    Real-time inference for intraday setups.
    """
    
    def __init__(self):
        self.models = {}
        self.load_models()
    
    def load_models(self):
        """Load trained models"""
        model_dir = 'models/intraday'
        
        if os.path.exists(f'{model_dir}/orb_xgboost.pkl'):
            self.models['ORB'] = joblib.load(f'{model_dir}/orb_xgboost.pkl')
            print("✅ Loaded ORB model")
        
        if os.path.exists(f'{model_dir}/vwap_reversion_xgboost.pkl'):
            self.models['VWAP_REVERSION'] = joblib.load(f'{model_dir}/vwap_reversion_xgboost.pkl')
            print("✅ Loaded VWAP Reversion model")
    
    def prepare_features(self, df, setup_type):
        """
        Prepare features for prediction.
        
        Args:
            df: DataFrame with calculated features
            setup_type: 'ORB' or 'VWAP_REVERSION'
        
        Returns:
            Feature matrix
        """
        feature_cols = [
            'distance_from_vwap_pct',
            'ema_9',
            'ema_21',
            'volume_spike_ratio',
            'intraday_volatility',
            'minutes_since_open'
        ]
        
        if setup_type == 'ORB':
            feature_cols.extend(['opening_range_high', 'opening_range_low'])
        
        # Get latest bar
        latest = df.iloc[-1]
        
        # Check if all features are available
        if any(pd.isna(latest[col]) for col in feature_cols):
            return None
        
        features = latest[feature_cols].values.reshape(1, -1)
        return features
    
    def generate_orb_signals(self, df):
        """
        Generate ORB signals.
        
        Args:
            df: DataFrame with features
        
        Returns:
            List of signals
        """
        if 'ORB' not in self.models:
            return []
        
        signals = []
        
        # Check for ORB conditions
        latest = df.iloc[-1]
        
        # Need opening range defined
        if pd.isna(latest['opening_range_high']):
            return []
        
        or_high = latest['opening_range_high']
        or_low = latest['opening_range_low']
        current_close = latest['close']
        volume_spike = latest['volume_spike_ratio']
        
        # Check for breakout
        if current_close > or_high and volume_spike >= 1.5:
            direction = 'LONG'
        elif current_close < or_low and volume_spike >= 1.5:
            direction = 'SHORT'
        else:
            return []
        
        # Get ML prediction
        features = self.prepare_features(df, 'ORB')
        if features is None:
            return []
        
        confidence = self.models['ORB'].predict_proba(features)[0, 1]
        
        # Only signal if confidence > 60%
        if confidence < 0.6:
            return []
        
        # Calculate entry, stop, target
        entry_price = current_close
        
        if direction == 'LONG':
            stop_loss = (or_high + or_low) / 2  # Midpoint of range
            target = entry_price + (entry_price - stop_loss) * 1.5  # 1.5R
        else:
            stop_loss = (or_high + or_low) / 2
            target = entry_price - (stop_loss - entry_price) * 1.5
        
        signals.append({
            'symbol': latest.get('symbol', 'UNKNOWN'),
            'timestamp': latest.get('timestamp', datetime.now()).strftime('%Y-%m-%d %H:%M:%S'),
            'setup_type': 'ORB',
            'signal': direction,
            'entry_price': float(entry_price),
            'current_price': float(current_close),
            'target': float(target),
            'stop_loss': float(stop_loss),
            'confidence': float(confidence),
            'status': 'ACTIVE'
        })
        
        return signals
    
    def generate_vwap_signals(self, df):
        """
        Generate VWAP reversion signals.
        
        Args:
            df: DataFrame with features
        
        Returns:
            List of signals
        """
        if 'VWAP_REVERSION' not in self.models:
            return []
        
        signals = []
        
        latest = df.iloc[-1]
        
        distance_pct = latest['distance_from_vwap_pct']
        volume_spike = latest['volume_spike_ratio']
        
        # Check for stretch conditions
        if distance_pct >= 0.4 and volume_spike <= 0.8:
            direction = 'SHORT'  # Price above VWAP, expect reversion down
        elif distance_pct <= -0.4 and volume_spike <= 0.8:
            direction = 'LONG'  # Price below VWAP, expect reversion up
        else:
            return []
        
        # Get ML prediction
        features = self.prepare_features(df, 'VWAP_REVERSION')
        if features is None:
            return []
        
        confidence = self.models['VWAP_REVERSION'].predict_proba(features)[0, 1]
        
        # Only signal if confidence > 60%
        if confidence < 0.6:
            return []
        
        # Calculate entry, stop, target
        entry_price = latest['close']
        vwap = latest['vwap']
        
        if direction == 'LONG':
            stop_loss = entry_price - abs(entry_price - vwap) * 1.2
            target = vwap  # Target is VWAP
        else:
            stop_loss = entry_price + abs(entry_price - vwap) * 1.2
            target = vwap
        
        signals.append({
            'symbol': latest.get('symbol', 'UNKNOWN'),
            'timestamp': latest.get('timestamp', datetime.now()).strftime('%Y-%m-%d %H:%M:%S'),
            'setup_type': 'VWAP_REVERSION',
            'signal': direction,
            'entry_price': float(entry_price),
            'current_price': float(entry_price),
            'target': float(target),
            'stop_loss': float(stop_loss),
            'confidence': float(confidence),
            'status': 'ACTIVE'
        })
        
        return signals
    
    def generate_all_signals(self, df):
        """
        Generate all intraday signals.
        
        Args:
            df: DataFrame with features
        
        Returns:
            List of all signals
        """
        signals = []
        
        # ORB signals
        signals.extend(self.generate_orb_signals(df))
        
        # VWAP signals
        signals.extend(self.generate_vwap_signals(df))
        
        return signals


def demo_inference():
    """Demo of real-time inference"""
    from src.intraday.data_ingestion.yahoo_intraday import YahooIntradayClient
    from src.intraday.features import IntradayFeatureEngine
    
    print("=" * 60)
    print("INTRADAY INFERENCE DEMO")
    print("=" * 60)
    
    # Initialize
    inference = IntradayInference()
    client = YahooIntradayClient()
    feature_engine = IntradayFeatureEngine()
    
    # Fetch data
    print("\n📊 Fetching RELIANCE intraday data...")
    df = client.fetch_5min_data('RELIANCE.NS', days_back=2)
    
    if df.empty:
        print("No data fetched")
        return
    
    # Generate features
    print("🔧 Generating features...")
    df = feature_engine.generate_features(df)
    df['symbol'] = 'RELIANCE.NS'
    
    # Generate signals
    print("🤖 Generating ML-based signals...")
    signals = inference.generate_all_signals(df)
    
    print(f"\n✅ Generated {len(signals)} signals:\n")
    for signal in signals:
        print(f"  {signal['setup_type']} {signal['signal']}")
        print(f"    Entry: ₹{signal['entry_price']:.2f}")
        print(f"    Target: ₹{signal['target']:.2f}")
        print(f"    Stop: ₹{signal['stop_loss']:.2f}")
        print(f"    Confidence: {signal['confidence']*100:.1f}%")
        print()


if __name__ == "__main__":
    demo_inference()
