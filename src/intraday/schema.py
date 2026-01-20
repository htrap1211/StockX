"""
Intraday Database Schema Setup

Creates tables for NSE intraday trading system:
- 5-minute OHLCV data
- Daily universe selection
- Trade signals (for backtesting)
"""

from sqlalchemy import create_engine, Column, String, Float, BigInteger, DateTime, Boolean, Integer, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

Base = declarative_base()


class IntradayPrice(Base):
    """5-minute intraday OHLCV data"""
    __tablename__ = 'nse_intraday_prices'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(BigInteger)
    
    __table_args__ = (
        # Composite index for fast queries
        {'schema': 'public'}
    )


class DailyUniverse(Base):
    """Daily selected universe (20-30 stocks)"""
    __tablename__ = 'intraday_daily_universe'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(DateTime, nullable=False, index=True)
    symbol = Column(String(20), nullable=False)
    selection_reason = Column(Text)  # Why this stock was selected
    atr_percentile = Column(Float)
    gap_pct = Column(Float)
    preopen_volume_spike = Column(Float)
    has_news = Column(Boolean, default=False)


class IntradaySignal(Base):
    """Generated signals (for backtesting only)"""
    __tablename__ = 'intraday_signals'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    symbol = Column(String(20), nullable=False)
    setup_type = Column(String(20))  # 'ORB' or 'VWAP_REVERSION'
    signal_type = Column(String(10))  # 'LONG' or 'SHORT'
    confidence = Column(Float)
    confidence_percentile = Column(Float)
    entry_price = Column(Float)
    stop_loss = Column(Float)
    target = Column(Float)
    
    # Filtering flags
    passed_volume_filter = Column(Boolean)
    passed_spread_filter = Column(Boolean)
    passed_regime_filter = Column(Boolean)
    
    # Outcome (for backtesting)
    executed = Column(Boolean, default=False)
    exit_price = Column(Float)
    exit_reason = Column(String(20))  # 'TARGET', 'STOP', 'TIME'
    pnl_pct = Column(Float)


def create_intraday_tables(database_url='postgresql://localhost/stock_db'):
    """
    Create all intraday tables.
    
    Args:
        database_url: PostgreSQL connection string
    """
    engine = create_engine(database_url)
    
    # Create tables
    Base.metadata.create_all(engine)
    
    print("✅ Intraday tables created:")
    print("  - nse_intraday_prices")
    print("  - intraday_daily_universe")
    print("  - intraday_signals")
    
    # Create indexes
    with engine.connect() as conn:
        # Composite index for fast time-series queries
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_intraday_symbol_timestamp 
            ON nse_intraday_prices(symbol, timestamp DESC);
        """)
        
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_universe_date 
            ON intraday_daily_universe(date DESC);
        """)
        
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_signals_timestamp 
            ON intraday_signals(timestamp DESC);
        """)
        
        conn.commit()
    
    print("✅ Indexes created")


if __name__ == "__main__":
    # Test with local database
    create_intraday_tables()
