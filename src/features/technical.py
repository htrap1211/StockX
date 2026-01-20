import pandas as pd
import numpy as np

def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index (RSI)"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    # Calculate RS
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """Moving Average Convergence Divergence (MACD)"""
    # Use EWM for MACD as per standard
    exp1 = series.ewm(span=fast, adjust=False).mean()
    exp2 = series.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    
    return pd.DataFrame({
        f'MACD_{fast}_{slow}': macd,
        f'MACD_signal_{signal}': signal_line,
        f'MACD_hist': histogram
    })

def calculate_bollinger_bands(series: pd.Series, window: int = 20, num_std: int = 2) -> pd.DataFrame:
    """Bollinger Bands"""
    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()
    
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    
    # %B = (Price - Lower Band) / (Upper Band - Lower Band)
    percent_b = (series - lower_band) / (upper_band - lower_band)
    
    # Band Width = (Upper Band - Lower Band) / Middle Band
    band_width = (upper_band - lower_band) / rolling_mean

    return pd.DataFrame({
        f'BB_upper_{window}': upper_band,
        f'BB_lower_{window}': lower_band,
        f'BB_mid_{window}': rolling_mean,
        f'BB_percent_b': percent_b,
        f'BB_width': band_width
    })

def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Average True Range (ATR)"""
    # TR = Max(High - Low, |High - PrevClose|, |Low - PrevClose|)
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=window).mean()
    return atr

def calculate_ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential Moving Average"""
    return series.ewm(span=span, adjust=False).mean()

def calculate_momentum(series: pd.Series, period: int) -> pd.Series:
    """Momentum (Price change)"""
    return series.diff(period)
