from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Index
from src.shared.database import Base

class Ticker(Base):
    __tablename__ = "tickers"

    symbol = Column(String, primary_key=True, index=True)
    market = Column(String, nullable=False)  # 'US' or 'IN'
    name = Column(String, nullable=True)
    sector = Column(String, nullable=True)
    industry = Column(String, nullable=True)
    is_active = Column(Boolean, default=True)

class MarketDataDaily(Base):
    __tablename__ = "market_data_daily"

    symbol = Column(String, primary_key=True)  # Composite PK part 1
    date = Column(DateTime, primary_key=True)   # Composite PK part 2
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    adjusted_close = Column(Float, nullable=True)

    __table_args__ = (
        Index('idx_market_data_symbol_date', 'symbol', 'date', unique=True),
        # TimescaleDB usually requires time to be part of the primary key or partitioning key
    )

class CorporateAction(Base):
    __tablename__ = "corporate_actions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String, index=True, nullable=False)
    date = Column(DateTime, nullable=False)
    action_type = Column(String, nullable=False)  # 'DIVIDEND', 'SPLIT'
    value = Column(Float, nullable=False)
