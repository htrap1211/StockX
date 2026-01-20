"""
Market Regime Detection Module

Implements professional regime classification:
1. Volatility Regime (Low/Medium/High)
2. Trend Regime (Uptrend/Downtrend)
3. Market Breadth (Strong/Weak) - Optional

Based on systematic fund best practices.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import Tuple, Dict
from datetime import datetime, timedelta


class RegimeDetector:
    """
    Detects market regimes for decision-sound trading.
    
    Regimes change strategy behavior:
    - Momentum works in uptrends
    - Mean reversion works in ranges
    - Risk explodes in high volatility
    """
    
    def __init__(self, market='US'):
        self.market = market
        self.index_symbol = 'SPY' if market == 'US' else '^NSEI'  # S&P 500 or NIFTY 50
        self.vix_symbol = '^VIX' if market == 'US' else '^INDIAVIX'
    
    def fetch_index_data(self, days=500) -> pd.DataFrame:
        """Fetch index data for regime calculation"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        try:
            data = yf.download(self.index_symbol, start=start_date, end=end_date, progress=False)
            # Handle multi-index columns from yfinance
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            data.columns = [c.lower() for c in data.columns]
            return data
        except Exception as e:
            print(f"Error fetching index data: {e}")
            return pd.DataFrame()
    
    def fetch_vix_data(self, days=500) -> pd.DataFrame:
        """Fetch VIX data for volatility regime"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        try:
            data = yf.download(self.vix_symbol, start=start_date, end=end_date, progress=False)
            return data['Close'] if not data.empty else pd.Series()
        except Exception as e:
            print(f"VIX not available, using index volatility: {e}")
            return pd.Series()
    
    def calculate_volatility_regime(self, index_data: pd.DataFrame, window=252) -> str:
        """
        Calculate volatility regime using percentile ranking.
        
        Logic:
        - Calculate ATR or realized volatility
        - Rank against 252-day history
        - Classify into Low/Medium/High
        
        Returns: 'LOW' | 'MEDIUM' | 'HIGH'
        """
        # Try VIX first
        vix = self.fetch_vix_data(days=window)
        
        if not vix.empty and len(vix) > 20:
            current_vix = float(vix.iloc[-1])
            # Calculate percentile: what % of historical values are <= current
            percentile = float((vix <= current_vix).sum()) / float(len(vix))
        else:
            # Fallback: Use index realized volatility
            if 'close' not in index_data.columns or len(index_data) < 50:
                return 'MEDIUM'  # Default if insufficient data
                
            returns = index_data['close'].pct_change()
            rolling_vol = returns.rolling(20).std() * np.sqrt(252)  # Annualized
            rolling_vol = rolling_vol.dropna()
            
            if len(rolling_vol) < 20:
                return 'MEDIUM'
            
            current_vol = float(rolling_vol.iloc[-1])
            # Calculate percentile
            percentile = float((rolling_vol <= current_vol).sum()) / float(len(rolling_vol))
        
        # Classify - percentile is now guaranteed to be a scalar float
        if percentile < 0.30:
            return 'LOW'
        elif percentile < 0.70:
            return 'MEDIUM'
        else:
            return 'HIGH'
    
    def calculate_trend_regime(self, index_data: pd.DataFrame) -> str:
        """
        Calculate trend regime using EMA crossover.
        
        Logic:
        - EMA 50 > EMA 200 → Uptrend
        - EMA 50 < EMA 200 → Downtrend
        
        Returns: 'UPTREND' | 'DOWNTREND'
        """
        ema_50 = index_data['close'].ewm(span=50, adjust=False).mean().iloc[-1]
        ema_200 = index_data['close'].ewm(span=200, adjust=False).mean().iloc[-1]
        
        return 'UPTREND' if ema_50 > ema_200 else 'DOWNTREND'
    
    def calculate_breadth_regime(self, universe_data: pd.DataFrame) -> str:
        """
        Calculate market breadth (optional).
        
        Logic:
        - % of stocks above EMA 50
        - > 50% → Strong breadth
        - < 50% → Weak breadth
        
        Returns: 'STRONG' | 'WEAK'
        """
        if universe_data.empty:
            return 'UNKNOWN'
        
        # Calculate EMA 50 for each stock
        ema_50 = universe_data.groupby('symbol')['close'].transform(
            lambda x: x.ewm(span=50, adjust=False).mean()
        )
        
        # Check if current price > EMA 50
        above_ema = (universe_data['close'] > ema_50).mean()
        
        return 'STRONG' if above_ema > 0.5 else 'WEAK'
    
    def get_current_regime(self) -> Dict[str, str]:
        """
        Get current market regime across all dimensions.
        
        Returns:
        {
            'volatility': 'LOW' | 'MEDIUM' | 'HIGH',
            'trend': 'UPTREND' | 'DOWNTREND',
            'breadth': 'STRONG' | 'WEAK' | 'UNKNOWN',
            'timestamp': datetime
        }
        """
        index_data = self.fetch_index_data()
        
        if index_data.empty:
            return {
                'volatility': 'UNKNOWN',
                'trend': 'UNKNOWN',
                'breadth': 'UNKNOWN',
                'timestamp': datetime.now()
            }
        
        vol_regime = self.calculate_volatility_regime(index_data)
        trend_regime = self.calculate_trend_regime(index_data)
        
        return {
            'volatility': vol_regime,
            'trend': trend_regime,
            'breadth': 'UNKNOWN',  # Requires universe data
            'timestamp': datetime.now(),
            'market': self.market
        }
    
    def encode_regime_features(self, regime: Dict[str, str]) -> Dict[str, int]:
        """
        Encode regimes as numeric features for ML model.
        
        Returns:
        {
            'vol_regime_low': 0/1,
            'vol_regime_medium': 0/1,
            'vol_regime_high': 0/1,
            'trend_regime_up': 0/1,
            'trend_regime_down': 0/1
        }
        """
        return {
            'vol_regime_low': 1 if regime['volatility'] == 'LOW' else 0,
            'vol_regime_medium': 1 if regime['volatility'] == 'MEDIUM' else 0,
            'vol_regime_high': 1 if regime['volatility'] == 'HIGH' else 0,
            'trend_regime_up': 1 if regime['trend'] == 'UPTREND' else 0,
            'trend_regime_down': 1 if regime['trend'] == 'DOWNTREND' else 0
        }


def should_trade_in_regime(regime: Dict[str, str], historical_sharpe: Dict[Tuple[str, str], float]) -> bool:
    """
    Determine if we should trade given current regime.
    
    Logic:
    - Only trade if historical Sharpe in this regime > threshold
    - Flat months > losing months
    
    Args:
        regime: Current regime dict
        historical_sharpe: Dict mapping (trend, vol) -> Sharpe ratio
    
    Returns:
        bool: True if should trade
    """
    regime_key = (regime['trend'], regime['volatility'])
    sharpe = historical_sharpe.get(regime_key, 0.0)
    
    # Only trade if historically profitable
    return sharpe > 0.5


if __name__ == "__main__":
    # Test regime detection
    print("Testing Regime Detection...")
    
    # US Market
    us_detector = RegimeDetector(market='US')
    us_regime = us_detector.get_current_regime()
    print(f"\nUS Market Regime:")
    print(f"  Volatility: {us_regime['volatility']}")
    print(f"  Trend: {us_regime['trend']}")
    print(f"  Timestamp: {us_regime['timestamp']}")
    
    # India Market
    in_detector = RegimeDetector(market='IN')
    in_regime = in_detector.get_current_regime()
    print(f"\nIndia Market Regime:")
    print(f"  Volatility: {in_regime['volatility']}")
    print(f"  Trend: {in_regime['trend']}")
    print(f"  Timestamp: {in_regime['timestamp']}")
    
    # Encoded features
    encoded = us_detector.encode_regime_features(us_regime)
    print(f"\nEncoded Features: {encoded}")
