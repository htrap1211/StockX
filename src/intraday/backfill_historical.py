"""
Historical Data Backfill for Intraday Models

Backfills 6 months of 5-minute data for model training.
Generates labels for ORB and VWAP setups.

Target: 10,000+ labeled setups for production-grade models.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add parent to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.intraday.data_ingestion.yahoo_intraday import YahooIntradayClient
from src.intraday.features import IntradayFeatureEngine
from src.intraday.setups.orb import ORBSetup
from src.intraday.setups.vwap_reversion import VWAPReversionSetup


class HistoricalBackfill:
    """
    Backfill historical intraday data and generate labels.
    """
    
    def __init__(self):
        self.client = YahooIntradayClient()
        self.feature_engine = IntradayFeatureEngine()
        self.orb_setup = ORBSetup()
        self.vwap_setup = VWAPReversionSetup()
    
    def backfill_symbol(self, symbol: str, months: int = 6):
        """
        Backfill data for one symbol.
        
        Args:
            symbol: Stock symbol (e.g., 'RELIANCE.NS')
            months: Number of months to backfill
        
        Returns:
            DataFrame with features and labels
        """
        print(f"\n{'='*60}")
        print(f"Backfilling {symbol} - {months} months")
        print(f"{'='*60}")
        
        all_data = []
        
        # Fetch data in chunks (Yahoo limits to 7 days per request)
        end_date = datetime.now()
        
        for chunk in range(months * 4):  # ~4 weeks per month
            start_date = end_date - timedelta(days=7)
            
            print(f"Fetching {start_date.date()} to {end_date.date()}...")
            
            df = self.client.fetch_5min_data(symbol, days_back=7)
            
            if not df.empty:
                all_data.append(df)
            
            end_date = start_date
        
        if not all_data:
            print(f"No data fetched for {symbol}")
            return pd.DataFrame()
        
        # Combine all chunks
        df_combined = pd.concat(all_data, ignore_index=True)
        df_combined = df_combined.drop_duplicates(subset=['timestamp'])
        df_combined = df_combined.sort_values('timestamp')
        
        print(f"Total bars fetched: {len(df_combined)}")
        
        # Generate features
        print("Generating features...")
        df_combined = self.feature_engine.generate_features(df_combined)
        
        # Generate labels
        print("Labeling ORB setups...")
        df_combined = self.orb_setup.generate_labels(df_combined)
        
        print("Labeling VWAP setups...")
        df_combined = self.vwap_setup.generate_labels(df_combined)
        
        # Stats
        orb_signals = (df_combined['orb_signal'] == 1).sum()
        orb_success = (df_combined['orb_label'] == 1).sum()
        vwap_signals = (df_combined['vwap_signal'] == 1).sum()
        vwap_success = (df_combined['vwap_label'] == 1).sum()
        
        print(f"\n✅ Backfill complete:")
        print(f"   Total bars: {len(df_combined)}")
        print(f"   ORB signals: {orb_signals} ({orb_success} successful)")
        print(f"   VWAP signals: {vwap_signals} ({vwap_success} successful)")
        
        return df_combined
    
    def backfill_universe(self, symbols: list, months: int = 6):
        """
        Backfill entire universe.
        
        Args:
            symbols: List of symbols
            months: Number of months
        
        Returns:
            Combined DataFrame
        """
        all_data = []
        
        for symbol in symbols:
            df = self.backfill_symbol(symbol, months)
            if not df.empty:
                all_data.append(df)
        
        if not all_data:
            return pd.DataFrame()
        
        df_combined = pd.concat(all_data, ignore_index=True)
        
        print(f"\n{'='*60}")
        print(f"UNIVERSE BACKFILL COMPLETE")
        print(f"{'='*60}")
        print(f"Total symbols: {len(symbols)}")
        print(f"Total bars: {len(df_combined)}")
        print(f"ORB setups: {(df_combined['orb_signal'] == 1).sum()}")
        print(f"VWAP setups: {(df_combined['vwap_signal'] == 1).sum()}")
        
        return df_combined
    
    def save_to_database(self, df: pd.DataFrame):
        """Save labeled data to database"""
        print("\nSaving to database...")
        
        # Save to CSV for now (database save can be added)
        output_file = 'data/intraday_labeled_data.csv'
        df.to_csv(output_file, index=False)
        
        print(f"✅ Saved to {output_file}")
        print(f"   Size: {len(df)} rows")


def run_backfill():
    """Run historical backfill"""
    print("=" * 60)
    print("HISTORICAL INTRADAY DATA BACKFILL")
    print("=" * 60)
    print()
    print("This will:")
    print("1. Fetch 6 months of 5-min data")
    print("2. Calculate features (VWAP, EMA, volume, etc.)")
    print("3. Label ORB and VWAP setups")
    print("4. Save to data/intraday_labeled_data.csv")
    print()
    print("Target: 10,000+ labeled setups")
    print("Time: ~30 minutes")
    print()
    
    # NIFTY 50 subset for demo
    symbols = [
        'RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'ICICIBANK.NS',
        'HINDUNILVR.NS', 'ITC.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'KOTAKBANK.NS'
    ]
    
    print(f"Symbols: {len(symbols)}")
    print(f"Expected bars: ~{len(symbols) * 6 * 20 * 75:,} (6 months, 20 days/month, 75 bars/day)")
    print()
    
    proceed = input("Proceed with backfill? (yes/no): ")
    
    if proceed.lower() != 'yes':
        print("Backfill cancelled.")
        return
    
    # Run backfill
    backfill = HistoricalBackfill()
    df = backfill.backfill_universe(symbols, months=6)
    
    if not df.empty:
        backfill.save_to_database(df)
        
        print("\n" + "=" * 60)
        print("✅ BACKFILL COMPLETE")
        print("=" * 60)
        print(f"\nNext step: Retrain models")
        print(f"Run: python src/intraday/models/train.py --data data/intraday_labeled_data.csv")
    else:
        print("\n❌ Backfill failed - no data collected")


if __name__ == "__main__":
    run_backfill()
