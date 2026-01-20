"""
Daily Universe Selector (Rule-Based)

Runs at 09:10 AM IST to select 20-30 stocks for intraday trading.

Selection criteria (need at least 2):
1. NIFTY 50 / NEXT 50 membership
2. Daily ATR in top 30%
3. Pre-open volume spike
4. Gap ≥ 0.5%
5. Earnings/news today
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict


class UniverseSelector:
    """
    Rule-based daily stock selector for intraday trading.
    """
    
    def __init__(self):
        # NIFTY 50 stocks (subset for now)
        self.nifty50 = [
            "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
            "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS",
            "LT.NS", "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS", "SUNPHARMA.NS",
            "TITAN.NS", "BAJFINANCE.NS", "ULTRACEMCO.NS", "NESTLEIND.NS", "WIPRO.NS"
        ]
        
        # NIFTY NEXT 50 (subset)
        self.nifty_next50 = [
            "ADANIPORTS.NS", "ADANIENT.NS", "APOLLOHOSP.NS", "BAJAJ-AUTO.NS",
            "BRITANNIA.NS", "CIPLA.NS", "DIVISLAB.NS", "DRREDDY.NS", "EICHERMOT.NS",
            "GRASIM.NS"
        ]
        
        self.universe_pool = self.nifty50 + self.nifty_next50
    
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(period).mean().iloc[-1]
        
        return atr
    
    def check_gap(self, df: pd.DataFrame) -> float:
        """
        Check for gap up/down.
        
        Returns:
            Gap percentage
        """
        if len(df) < 2:
            return 0.0
        
        prev_close = df['close'].iloc[-2]
        today_open = df['open'].iloc[-1]
        
        gap_pct = ((today_open - prev_close) / prev_close) * 100
        
        return gap_pct
    
    def check_volume_spike(self, df: pd.DataFrame, lookback: int = 20) -> float:
        """
        Check for pre-open volume spike.
        
        Returns:
            Volume spike ratio (current / avg)
        """
        if len(df) < lookback:
            return 1.0
        
        avg_volume = df['volume'].iloc[-lookback:-1].mean()
        current_volume = df['volume'].iloc[-1]
        
        if avg_volume == 0:
            return 1.0
        
        spike_ratio = current_volume / avg_volume
        
        return spike_ratio
    
    def select_daily_universe(self, 
                              market_data: Dict[str, pd.DataFrame],
                              date: datetime) -> List[Dict]:
        """
        Select 20-30 stocks for the day.
        
        Args:
            market_data: Dict of {symbol: daily_df}
            date: Trading date
        
        Returns:
            List of selected stocks with reasons
        """
        selected = []
        
        # Calculate ATR for all stocks
        atr_values = {}
        for symbol, df in market_data.items():
            if len(df) >= 14:
                atr_values[symbol] = self.calculate_atr(df)
        
        # ATR percentile threshold (top 30%)
        if atr_values:
            atr_threshold = np.percentile(list(atr_values.values()), 70)
        else:
            atr_threshold = 0
        
        # Evaluate each stock
        for symbol in self.universe_pool:
            if symbol not in market_data:
                continue
            
            df = market_data[symbol]
            if len(df) < 20:
                continue
            
            criteria_met = []
            
            # Criterion 1: NIFTY membership (always true)
            criteria_met.append("NIFTY_MEMBER")
            
            # Criterion 2: ATR in top 30%
            atr = atr_values.get(symbol, 0)
            if atr >= atr_threshold:
                criteria_met.append(f"HIGH_ATR({atr:.2f})")
            
            # Criterion 3: Volume spike
            vol_spike = self.check_volume_spike(df)
            if vol_spike >= 1.5:
                criteria_met.append(f"VOL_SPIKE({vol_spike:.1f}x)")
            
            # Criterion 4: Gap
            gap_pct = self.check_gap(df)
            if abs(gap_pct) >= 0.5:
                criteria_met.append(f"GAP({gap_pct:+.1f}%)")
            
            # Criterion 5: News (placeholder - would need news API)
            # has_news = check_news(symbol, date)
            
            # Select if at least 2 criteria met
            if len(criteria_met) >= 2:
                selected.append({
                    'symbol': symbol,
                    'date': date,
                    'criteria_met': criteria_met,
                    'atr': atr,
                    'gap_pct': gap_pct,
                    'vol_spike': vol_spike
                })
        
        # Limit to 30 stocks (prioritize by number of criteria)
        selected = sorted(selected, key=lambda x: len(x['criteria_met']), reverse=True)[:30]
        
        return selected


def example_usage():
    """Example of daily universe selection"""
    
    selector = UniverseSelector()
    
    # Simulate market data (would come from database)
    market_data = {}
    
    # For demo, create dummy data
    dates = pd.date_range('2026-01-01', '2026-01-20', freq='D')
    
    for symbol in selector.universe_pool[:5]:  # Test with 5 stocks
        df = pd.DataFrame({
            'date': dates,
            'open': np.random.randn(len(dates)) * 10 + 1300,
            'high': np.random.randn(len(dates)) * 10 + 1310,
            'low': np.random.randn(len(dates)) * 10 + 1290,
            'close': np.random.randn(len(dates)) * 10 + 1300,
            'volume': np.random.randint(1000000, 5000000, len(dates))
        })
        market_data[symbol] = df
    
    # Select universe
    today = datetime(2026, 1, 20)
    selected = selector.select_daily_universe(market_data, today)
    
    print(f"Selected {len(selected)} stocks for {today.date()}:\n")
    for stock in selected:
        print(f"{stock['symbol']:20} | Criteria: {', '.join(stock['criteria_met'])}")


if __name__ == "__main__":
    example_usage()
