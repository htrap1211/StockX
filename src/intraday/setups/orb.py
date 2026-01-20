"""
Opening Range Breakout (ORB) Setup

Entry Conditions:
- First 15-minute range formed
- Price breaks range with volume spike
- NIFTY trend aligns

Label:
- 1 if +0.25% move within next 3 candles (15 minutes)
- 0 otherwise
"""

import pandas as pd
import numpy as np
from typing import Tuple


class ORBSetup:
    """
    Opening Range Breakout setup detector and labeler.
    """
    
    def __init__(self, 
                 target_pct=0.25,
                 lookforward_bars=3,
                 volume_spike_threshold=1.5):
        """
        Args:
            target_pct: Target return % for label=1
            lookforward_bars: Bars to check for target
            volume_spike_threshold: Min volume spike for valid breakout
        """
        self.target_pct = target_pct
        self.lookforward_bars = lookforward_bars
        self.volume_spike_threshold = volume_spike_threshold
    
    def detect_breakout(self, df: pd.DataFrame, idx: int) -> Tuple[bool, str]:
        """
        Detect if ORB breakout occurred at given index.
        
        Args:
            df: DataFrame with features
            idx: Current bar index
        
        Returns:
            (is_breakout, direction)
        """
        # Need opening range defined
        if pd.isna(df.loc[idx, 'opening_range_high']):
            return False, None
        
        or_high = df.loc[idx, 'opening_range_high']
        or_low = df.loc[idx, 'opening_range_low']
        
        current_close = df.loc[idx, 'close']
        volume_spike = df.loc[idx, 'volume_spike_ratio']
        
        # Check for breakout with volume
        if current_close > or_high and volume_spike >= self.volume_spike_threshold:
            return True, 'LONG'
        elif current_close < or_low and volume_spike >= self.volume_spike_threshold:
            return True, 'SHORT'
        
        return False, None
    
    def calculate_forward_return(self, df: pd.DataFrame, idx: int, direction: str) -> float:
        """
        Calculate forward return over next N bars.
        
        Args:
            df: DataFrame
            idx: Entry index
            direction: 'LONG' or 'SHORT'
        
        Returns:
            Forward return %
        """
        if idx + self.lookforward_bars >= len(df):
            return 0.0
        
        entry_price = df.loc[idx, 'close']
        
        # Get max/min over next bars
        future_bars = df.iloc[idx+1:idx+1+self.lookforward_bars]
        
        if direction == 'LONG':
            best_price = future_bars['high'].max()
            return_pct = ((best_price - entry_price) / entry_price) * 100
        else:  # SHORT
            best_price = future_bars['low'].min()
            return_pct = ((entry_price - best_price) / entry_price) * 100
        
        return return_pct
    
    def generate_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate labels for ORB setup.
        
        Args:
            df: DataFrame with features
        
        Returns:
            DataFrame with labels added
        """
        df = df.copy()
        df = df.reset_index(drop=True)
        
        # Initialize columns
        df['orb_signal'] = 0
        df['orb_direction'] = None
        df['orb_forward_return'] = 0.0
        df['orb_label'] = 0
        
        # Scan for breakouts
        for idx in range(3, len(df) - self.lookforward_bars):  # Start after opening range
            is_breakout, direction = self.detect_breakout(df, idx)
            
            if is_breakout:
                forward_return = self.calculate_forward_return(df, idx, direction)
                
                df.loc[idx, 'orb_signal'] = 1
                df.loc[idx, 'orb_direction'] = direction
                df.loc[idx, 'orb_forward_return'] = forward_return
                
                # Label: 1 if target achieved
                if forward_return >= self.target_pct:
                    df.loc[idx, 'orb_label'] = 1
        
        return df


def example_usage():
    """Example of ORB labeling"""
    
    # Load sample data
    from src.intraday.data_ingestion.yahoo_intraday import YahooIntradayClient
    from src.intraday.features import IntradayFeatureEngine
    
    client = YahooIntradayClient()
    df = client.fetch_5min_data('RELIANCE.NS', days_back=3)
    
    if df.empty:
        print("No data")
        return
    
    # Generate features
    feature_engine = IntradayFeatureEngine()
    df = feature_engine.generate_features(df)
    
    # Generate ORB labels
    orb = ORBSetup(target_pct=0.25, lookforward_bars=3)
    df = orb.generate_labels(df)
    
    # Show signals
    signals = df[df['orb_signal'] == 1]
    
    print(f"Found {len(signals)} ORB breakouts\n")
    print(signals[['timestamp', 'close', 'orb_direction', 'orb_forward_return', 'orb_label']])
    
    # Stats
    if len(signals) > 0:
        success_rate = (signals['orb_label'] == 1).sum() / len(signals) * 100
        print(f"\nSuccess rate: {success_rate:.1f}%")


if __name__ == "__main__":
    example_usage()
