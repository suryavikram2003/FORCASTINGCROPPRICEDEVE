from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

MYSQL_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': 'vikram12345',
    'database': 'market_db'
}

DATABASE_URL = f"mysql+pymysql://{MYSQL_CONFIG['user']}:{MYSQL_CONFIG['password']}@{MYSQL_CONFIG['host']}/{MYSQL_CONFIG['database']}"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
