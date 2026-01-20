import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from typing import Dict, Any, Tuple

class ModelTrainer:
    def __init__(self, model_type: str = "xgboost"):
        self.model_type = model_type
        self.model = None
        
    def get_model(self):
        if self.model_type == "logistic":
            return LogisticRegression(class_weight='balanced', max_iter=1000)
        elif self.model_type == "xgboost":
            return xgb.XGBClassifier(
                objective='binary:logistic',
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                eval_metric='logloss'
                # tree_method='gpu_hist' # if GPU available
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def train_walk_forward(self, df: pd.DataFrame, target_col: str, feature_cols: list) -> Dict[str, Any]:
        """
        Performs Walk-Forward Validation.
        V1 Scheme:
        - Train Window: Initial 1 year (252 days).
        - Test Window: Next 1 month (20 days).
        - Step: Move forward by 20 days.
        - Retrain? Yes, expanding window training.
        """
        df = df.dropna(subset=feature_cols + [target_col]).sort_index()
        
        dates = df.index
        start_date = dates[0]
        end_date = dates[-1]
        
        # Initial train size: approx 1 year
        train_window_days = 252 
        test_window_days = 20
        
        current_date = start_date + pd.Timedelta(days=train_window_days)
        
        predictions = []
        actuals = []
        
        # Iterate
        while current_date + pd.Timedelta(days=test_window_days) <= end_date:
            train_mask = df.index < current_date
            test_mask = (df.index >= current_date) & (df.index < current_date + pd.Timedelta(days=test_window_days))
            
            X_train = df.loc[train_mask, feature_cols]
            y_train = df.loc[train_mask, target_col]
            
            X_test = df.loc[test_mask, feature_cols]
            y_test = df.loc[test_mask, target_col]
            
            if X_train.empty or X_test.empty:
                current_date += pd.Timedelta(days=test_window_days)
                continue
                
            # Train (or retrain)
            clf = self.get_model()
            clf.fit(X_train, y_train)
            
            # Predict
            preds = clf.predict(X_test)
            probs = clf.predict_proba(X_test)[:, 1]
            
            # Store
            # Standardize output: (date, actual, pred, prob)
            for idx, (p, prob, act) in enumerate(zip(preds, probs, y_test)):
                predictions.append({
                    'date': y_test.index[idx],
                    'actual': act,
                    'pred': p,
                    'prob': prob
                })
            
            # Only save the last model for final inference? 
            # Ideally we save the "latest" trained model for live inference.
            self.model = clf 
            
            # Move forward
            current_date += pd.Timedelta(days=test_window_days)
            
        results_df = pd.DataFrame(predictions)
        if results_df.empty:
            return {"status": "No predictions generated"}
            
        # Metrics
        y_true = results_df['actual']
        y_pred = results_df['pred']
        
        metrics = {
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_true, results_df['prob']) if len(results_df['prob'].unique()) > 1 else 0.5,
            "total_trades": len(results_df),
            "positive_preds": sum(y_pred)
        }
        
        return metrics, results_df
