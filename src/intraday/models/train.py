"""
Intraday Model Training Pipeline

Trains ML models for:
- ORB (Opening Range Breakout)
- VWAP Mean Reversion

Uses walk-forward validation and outputs trained models.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, roc_auc_score
import xgboost as xgb
import joblib
from datetime import datetime, timedelta
import os


class IntradayModelTrainer:
    """
    Train ML models for intraday setups.
    """
    
    def __init__(self, setup_type='ORB'):
        """
        Args:
            setup_type: 'ORB' or 'VWAP_REVERSION'
        """
        self.setup_type = setup_type
        self.models = {}
    
    def prepare_features(self, df):
        """
        Prepare feature matrix from labeled data.
        
        Args:
            df: DataFrame with features and labels
        
        Returns:
            X, y
        """
        # Feature columns
        feature_cols = [
            'distance_from_vwap_pct',
            'ema_9',
            'ema_21',
            'volume_spike_ratio',
            'intraday_volatility',
            'minutes_since_open'
        ]
        
        # Add setup-specific features
        if self.setup_type == 'ORB':
            feature_cols.extend(['opening_range_high', 'opening_range_low'])
            label_col = 'orb_label'
        else:  # VWAP_REVERSION
            label_col = 'vwap_label'
        
        # Filter valid rows
        df_clean = df.dropna(subset=feature_cols + [label_col])
        
        X = df_clean[feature_cols].values
        y = df_clean[label_col].values
        
        return X, y, df_clean.index
    
    def train_logistic_regression(self, X_train, y_train):
        """Train Logistic Regression model"""
        model = LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=42
        )
        model.fit(X_train, y_train)
        return model
    
    def train_xgboost(self, X_train, y_train):
        """Train XGBoost model with max_depth <= 4"""
        model = xgb.XGBClassifier(
            max_depth=4,
            n_estimators=100,
            learning_rate=0.1,
            scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1]),
            random_state=42
        )
        model.fit(X_train, y_train)
        return model
    
    def walk_forward_validation(self, df, n_splits=3):
        """
        Walk-forward validation for time series.
        
        Args:
            df: DataFrame with features and labels
            n_splits: Number of train/test splits
        
        Returns:
            Dict with model performance
        """
        X, y, indices = self.prepare_features(df)
        
        if len(X) < 100:
            print(f"Not enough data for {self.setup_type}: {len(X)} samples")
            return None
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        results = {
            'logistic': {'auc': [], 'accuracy': []},
            'xgboost': {'auc': [], 'accuracy': []}
        }
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Train Logistic Regression
            lr_model = self.train_logistic_regression(X_train, y_train)
            lr_pred = lr_model.predict(X_test)
            lr_proba = lr_model.predict_proba(X_test)[:, 1]
            
            results['logistic']['accuracy'].append((lr_pred == y_test).mean())
            if len(np.unique(y_test)) > 1:
                results['logistic']['auc'].append(roc_auc_score(y_test, lr_proba))
            
            # Train XGBoost
            xgb_model = self.train_xgboost(X_train, y_train)
            xgb_pred = xgb_model.predict(X_test)
            xgb_proba = xgb_model.predict_proba(X_test)[:, 1]
            
            results['xgboost']['accuracy'].append((xgb_pred == y_test).mean())
            if len(np.unique(y_test)) > 1:
                results['xgboost']['auc'].append(roc_auc_score(y_test, xgb_proba))
            
            print(f"Fold {fold+1}/{n_splits}:")
            print(f"  LR  - Acc: {results['logistic']['accuracy'][-1]:.3f}, AUC: {results['logistic']['auc'][-1]:.3f}")
            print(f"  XGB - Acc: {results['xgboost']['accuracy'][-1]:.3f}, AUC: {results['xgboost']['auc'][-1]:.3f}")
        
        return results
    
    def train_final_model(self, df, model_type='xgboost'):
        """
        Train final model on all data.
        
        Args:
            df: DataFrame with features and labels
            model_type: 'logistic' or 'xgboost'
        
        Returns:
            Trained model
        """
        X, y, _ = self.prepare_features(df)
        
        if model_type == 'logistic':
            model = self.train_logistic_regression(X, y)
        else:
            model = self.train_xgboost(X, y)
        
        print(f"\n✅ Trained final {model_type} model for {self.setup_type}")
        print(f"   Training samples: {len(X)}")
        print(f"   Positive samples: {y.sum()} ({y.sum()/len(y)*100:.1f}%)")
        
        return model
    
    def save_model(self, model, model_type='xgboost'):
        """Save trained model"""
        os.makedirs('models/intraday', exist_ok=True)
        filename = f'models/intraday/{self.setup_type.lower()}_{model_type}.pkl'
        joblib.dump(model, filename)
        print(f"✅ Model saved: {filename}")
        return filename


def train_all_intraday_models(data_file=None):
    """
    Train all intraday models.
    
    Args:
        data_file: Path to labeled data CSV (if None, uses demo data)
    """
    print("=" * 60)
    print("INTRADAY MODEL TRAINING PIPELINE")
    print("=" * 60)
    
    if data_file and os.path.exists(data_file):
        print(f"\n📊 Loading real historical data from {data_file}...")
        
        # Load data
        df = pd.DataFrame(joblib.load(data_file) if data_file.endswith('.pkl') else pd.read_csv(data_file))
        
        print(f"   Total rows: {len(df)}")
        
        # Split by setup type
        orb_data = df[df['orb_signal'].notna()].copy()
        vwap_data = df[df['vwap_signal'].notna()].copy()
        
        print(f"   ORB samples: {len(orb_data)}")
        print(f"   VWAP samples: {len(vwap_data)}")
        
    else:
        print("\n📊 Generating demo training data...")
        
        # Simulate labeled data
        n_samples = 500
        
        # ORB data
        orb_data = pd.DataFrame({
            'distance_from_vwap_pct': np.random.randn(n_samples) * 0.5,
            'ema_9': np.random.randn(n_samples) * 10 + 1400,
            'ema_21': np.random.randn(n_samples) * 10 + 1400,
            'volume_spike_ratio': np.random.uniform(0.5, 3.0, n_samples),
            'intraday_volatility': np.random.uniform(0.01, 0.05, n_samples),
            'minutes_since_open': np.random.randint(15, 360, n_samples),
            'opening_range_high': np.random.randn(n_samples) * 10 + 1410,
            'opening_range_low': np.random.randn(n_samples) * 10 + 1390,
            'orb_label': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
        })
        
        # VWAP data
        vwap_data = pd.DataFrame({
            'distance_from_vwap_pct': np.random.randn(n_samples) * 0.8,
            'ema_9': np.random.randn(n_samples) * 10 + 1400,
            'ema_21': np.random.randn(n_samples) * 10 + 1400,
            'volume_spike_ratio': np.random.uniform(0.3, 2.0, n_samples),
            'intraday_volatility': np.random.uniform(0.01, 0.05, n_samples),
            'minutes_since_open': np.random.randint(20, 360, n_samples),
            'vwap_label': np.random.choice([0, 1], n_samples, p=[0.65, 0.35])
        })
    
    # Train ORB models
    print("\n" + "=" * 60)
    print("TRAINING ORB MODELS")
    print("=" * 60)
    orb_trainer = IntradayModelTrainer(setup_type='ORB')
    orb_results = orb_trainer.walk_forward_validation(orb_data)
    orb_model = orb_trainer.train_final_model(orb_data, model_type='xgboost')
    orb_trainer.save_model(orb_model, model_type='xgboost')
    
    # Train VWAP models
    print("\n" + "=" * 60)
    print("TRAINING VWAP REVERSION MODELS")
    print("=" * 60)
    vwap_trainer = IntradayModelTrainer(setup_type='VWAP_REVERSION')
    vwap_results = vwap_trainer.walk_forward_validation(vwap_data)
    vwap_model = vwap_trainer.train_final_model(vwap_data, model_type='xgboost')
    vwap_trainer.save_model(vwap_model, model_type='xgboost')
    
    print("\n" + "=" * 60)
    print("✅ ALL MODELS TRAINED SUCCESSFULLY")
    print("=" * 60)
    print("\nModels saved:")
    print("  - models/intraday/orb_xgboost.pkl")
    print("  - models/intraday/vwap_reversion_xgboost.pkl")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train intraday ML models')
    parser.add_argument('--data', type=str, help='Path to labeled data CSV')
    args = parser.parse_args()
    
    train_all_intraday_models(data_file=args.data)
