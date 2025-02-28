from sqlalchemy import Column, String, Float, DateTime, Integer
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class MarketPrice(Base):
    __tablename__ = 'market_prices'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    state = Column(String(100))
    district = Column(String(100))
    market = Column(String(100))
    commodity = Column(String(100))
    variety = Column(String(100))
    arrival_date = Column(DateTime)
    min_price = Column(Float)
    max_price = Column(Float)
    modal_price = Column(Float)
    created_at = Column(DateTime)
    
    __table_args__ = {
        'mysql_engine': 'InnoDB',
        'mysql_charset': 'utf8mb4'
    }
