from flask import Flask, jsonify
from flask_cors import CORS
import requests
import logging
import pymysql
from datetime import datetime, timedelta
from sqlalchemy import Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine

# Configure logging to show only important information
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)
# Reduce SQLAlchemy logging
logging.getLogger('sqlalchemy.engine').setLevel(logging.WARNING)

# Database configuration
Base = declarative_base()
engine = create_engine('mysql+pymysql://root:vikram12345@localhost/market_db', echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Model definition
class MarketPrice(Base):
    __tablename__ = "market_prices"

    id = Column(Integer, primary_key=True, index=True)
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

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize database
Base.metadata.create_all(bind=engine)

class MarketAPI:
    BASE_URL = "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070"
    API_KEY = "579b464db66ec23bdd000001179646bab8714e4e704f4a2a752f7138"
    
    def get_market_data(self):
        today = datetime.now().strftime('%d/%m/%Y')
        yesterday = (datetime.now() - timedelta(days=1)).strftime('%d/%m/%Y')
        
        # Try today's data
        data = self.fetch_data_for_date(today)
        
        # If no data for today, get yesterday's data
        if not data.get('records'):
            logger.info(f"Getting yesterday's data: {yesterday}")
            data = self.fetch_data_for_date(yesterday)
        
        if data.get('records'):
            logger.info(f"Storing {len(data['records'])} records")
            self.store_in_db(data['records'])
            
        return data
    
    def fetch_data_for_date(self, date):
        headers = {
            'Accept': 'application/json'
        }
        
        params = {
            'api-key': self.API_KEY,
            'format': 'json',
            'filters[state]': 'Tamil Nadu',
            'filters[district]': 'Salem',
            'filters[arrival_date]': date,
            'limit': 1000
        }
        
        try:
            response = requests.get(
                self.BASE_URL,
                headers=headers,
                params=params
            )
            response.raise_for_status()
            data = response.json()
            
            if data and 'records' in data:
                logger.info(f"Successfully fetched {len(data['records'])} records for {date}")
                
                transformed_records = [{
                    'state': record['state'],
                    'district': record['district'],
                    'market': record['market'],
                    'commodity': record['commodity'],
                    'variety': record['variety'],
                    'arrival_date': record['arrival_date'],
                    'min_price': float(record['min_price']),
                    'max_price': float(record['max_price']),
                    'modal_price': float(record['modal_price'])
                } for record in data['records']]
                
                data['records'] = transformed_records
            
            return data
            
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Data fetch error: {str(e)}")
            return {'records': [], 'total': 0}
    
    def store_in_db(self, records):
        session = SessionLocal()
        try:
            stored_count = 0
            for record in records:
                arrival_date = datetime.strptime(record['arrival_date'], '%d/%m/%Y')
                market_price = MarketPrice(
                    state=record['state'],
                    district=record['district'],
                    market=record['market'],
                    commodity=record['commodity'],
                    variety=record['variety'],
                    arrival_date=arrival_date,
                    min_price=float(record['min_price']),
                    max_price=float(record['max_price']),
                    modal_price=float(record['modal_price']),
                    created_at=datetime.utcnow()
                )
                session.add(market_price)
                stored_count += 1
                
            session.commit()
            logger.info(f"✅ Successfully stored {stored_count} market price records")
            
        except Exception as e:
            logger.error(f"❌ Storage error: {str(e)}")
            session.rollback()
        finally:
            session.close()

@app.route('/api/market-data')
def get_market_data():
    api = MarketAPI()
    data = api.get_market_data()
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True)