import pandas as pd
import numpy as np

class LabelingEngine:
    def __init__(self):
        pass

    def add_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds target labels to the dataframe.
        
        Targets:
        1. Target_Swing: 1 if Max Return in next 20 days >= 5%, else 0? 
           User Spec: "Label = 1 if stock returns >= +5% within next 20 trading days"
           Interpretation: Is it just price at t+20 vs price at t? Or any time within?
           Usually "within" implies "at any point". But standard swing models often predict "exit at N days".
           Let's stick to strict: Price(t+20) / Price(t) - 1 >= 0.05 OR Price(any t+1..20) / Price(t) >= 1.05?
           "within next 20 trading days" usually sounds like "hits profit target".
           Let's implement: Max(High[t+1...t+20]) >= Close[t] * 1.05.
           BUT to prevent whipsaw, usually we also check if it hits stop loss.
           For Version 1, let's simplify to: Close(t+20) / Close(t) - 1 >= 0.05. 
           User Spec text: "Label = 1 if stock returns >= +5% within next 20 trading days". 
           Let's use the simple point-to-point return for stability in V1.
           
           Actually, "within" strongly suggests checking Highs. 
           Let's do: (Max(High[t+1 : t+20]) / Close[t]) >= 1.05.
        
        2. Target_LongTerm: 1 if stock outperforms index by >= 10% over 12 months.
           V1 Proxy: Absolute Return >= 15% in 252 days.
           Logic: Close(t+252) / Close(t) - 1 >= 0.15.
        """
        if df.empty:
            return df
        
        # Ensure sorted by date
        df = df.sort_index()
        
        close = df['close']
        high = df['high']
        
        # --- Swing Label ---
        # Look forward 20 days
        # We need the MAX high of the next 20 days.
        # shifting backwards! 
        # rolling max reversed?
        # easiest: df['high'][::-1].rolling(20).max()[::-1].shift(-1)
        # But let's use the simple Look Ahead approach via shift if we did point-to-point.
        
        # Calculating "Max High in next 20 days" efficiently
        indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=20)
        future_high_max = high.rolling(window=indexer).max().shift(-1) # shift -1 to act on t+1...t+20?
        # No, FixedForwardWindowIndexer at 't' includes 't'. 
        # We want t+1 to t+20.
        # So we can shift high by -1 first.
        
        next_20_highs = high.shift(-1).rolling(window=20, min_periods=20).max().shift(-19) 
        # Standard rolling with negative shift is cleaner? 
        # Let's stick to: Create a forward rolling max.
        
        # Alternative: Just use point-to-point for V1 as it's cleaner for 'Logistic Regression'.
        # "Target_Swing": Return(t, t+20) >= 0.05
        future_close_20 = close.shift(-20)
        ret_20 = (future_close_20 / close) - 1
        df['Target_Swing'] = (ret_20 >= 0.05).astype(int)
        
        # --- Long Term Label ---
        # Return(t, t+252) >= 0.15
        future_close_252 = close.shift(-252)
        ret_252 = (future_close_252 / close) - 1
        df['Target_LT'] = (ret_252 >= 0.15).astype(int)
        
        # Drop rows where targets are NaN (the last 20/252 days) happens during training setup usually, 
        # but we can leave them as NaNs or 0. Better to keep them as NaN so training drops them.
        
        return df
