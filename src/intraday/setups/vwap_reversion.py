"""
VWAP Mean Reversion Setup

Entry Conditions:
- Price stretched ≥ ±0.4% from VWAP
- Volume declining
- Market neutral

Label:
- 1 if 0.2% reversion toward VWAP within 4 candles (20 minutes)
- 0 otherwise
"""

import pandas as pd
import numpy as np
from typing import Tuple


class VWAPReversionSetup:
    """
    VWAP mean reversion setup detector and labeler.
    """
    
    def __init__(self,
                 stretch_threshold=0.4,
                 reversion_target=0.2,
                 lookforward_bars=4,
                 volume_decline_threshold=0.8):
        """
        Args:
            stretch_threshold: Min % distance from VWAP
            reversion_target: Target reversion % for label=1
            lookforward_bars: Bars to check for reversion
            volume_decline_threshold: Max volume ratio (declining)
        """
        self.stretch_threshold = stretch_threshold
        self.reversion_target = reversion_target
        self.lookforward_bars = lookforward_bars
        self.volume_decline_threshold = volume_decline_threshold
    
    def detect_stretch(self, df: pd.DataFrame, idx: int) -> Tuple[bool, str]:
        """
        Detect if price is stretched from VWAP.
        
        Args:
            df: DataFrame with features
            idx: Current bar index
        
        Returns:
            (is_stretched, direction)
        """
        distance_pct = df.loc[idx, 'distance_from_vwap_pct']
        volume_spike = df.loc[idx, 'volume_spike_ratio']
        
        # Check for stretch with declining volume
        if distance_pct >= self.stretch_threshold and volume_spike <= self.volume_decline_threshold:
            return True, 'SHORT'  # Price above VWAP, expect reversion down
        elif distance_pct <= -self.stretch_threshold and volume_spike <= self.volume_decline_threshold:
            return True, 'LONG'  # Price below VWAP, expect reversion up
        
        return False, None
    
    def calculate_reversion(self, df: pd.DataFrame, idx: int, direction: str) -> float:
        """
        Calculate reversion toward VWAP over next N bars.
        
        Args:
            df: DataFrame
            idx: Entry index
            direction: 'LONG' or 'SHORT'
        
        Returns:
            Reversion %
        """
        if idx + self.lookforward_bars >= len(df):
            return 0.0
        
        entry_price = df.loc[idx, 'close']
        vwap = df.loc[idx, 'vwap']
        
        # Get future bars
        future_bars = df.iloc[idx+1:idx+1+self.lookforward_bars]
        
        if direction == 'LONG':
            # Expect price to move up toward VWAP
            best_price = future_bars['high'].max()
            reversion_pct = ((best_price - entry_price) / entry_price) * 100
        else:  # SHORT
            # Expect price to move down toward VWAP
            best_price = future_bars['low'].min()
            reversion_pct = ((entry_price - best_price) / entry_price) * 100
        
        return reversion_pct
    
    def generate_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate labels for VWAP reversion setup.
        
        Args:
            df: DataFrame with features
        
        Returns:
            DataFrame with labels added
        """
        df = df.copy()
        df = df.reset_index(drop=True)
        
        # Initialize columns
        df['vwap_signal'] = 0
        df['vwap_direction'] = None
        df['vwap_reversion'] = 0.0
        df['vwap_label'] = 0
        
        # Scan for stretch conditions
        for idx in range(20, len(df) - self.lookforward_bars):  # Need history for volume
            is_stretched, direction = self.detect_stretch(df, idx)
            
            if is_stretched:
                reversion = self.calculate_reversion(df, idx, direction)
                
                df.loc[idx, 'vwap_signal'] = 1
                df.loc[idx, 'vwap_direction'] = direction
                df.loc[idx, 'vwap_reversion'] = reversion
                
                # Label: 1 if reversion target achieved
                if reversion >= self.reversion_target:
                    df.loc[idx, 'vwap_label'] = 1
        
        return df


def example_usage():
    """Example of VWAP reversion labeling"""
    
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
    
    # Generate VWAP reversion labels
    vwap_setup = VWAPReversionSetup(stretch_threshold=0.4, reversion_target=0.2)
    df = vwap_setup.generate_labels(df)
    
    # Show signals
    signals = df[df['vwap_signal'] == 1]
    
    print(f"Found {len(signals)} VWAP reversion setups\n")
    print(signals[['timestamp', 'close', 'vwap', 'distance_from_vwap_pct', 
                   'vwap_direction', 'vwap_reversion', 'vwap_label']])
    
    # Stats
    if len(signals) > 0:
        success_rate = (signals['vwap_label'] == 1).sum() / len(signals) * 100
        print(f"\nSuccess rate: {success_rate:.1f}%")


if __name__ == "__main__":
    example_usage()
