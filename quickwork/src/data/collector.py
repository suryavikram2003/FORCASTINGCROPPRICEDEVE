import pandas as pd
import requests
from datetime import datetime, timedelta

class DataCollector:
    def __init__(self, api_key):
        self.api_key = "579b464db66ec23bdd0000011c66a6e109084bad6d2eb0cb5aa45398"
        self.base_url = "https://data.gov.in/api/commodity-prices"

    def fetch_historical_data(self, commodity, start_date, end_date):
        params = {
            'commodity': commodity,
            'start_date': start_date,
            'end_date': end_date,
            'api_key': self.api_key
        }
        
        response = requests.get(self.base_url, params=params)
        data = response.json()
        
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])
        return df

    def preprocess_data(self, df):
        # Add features like moving averages, seasonal indicators
        df['price_ma7'] = df['price'].rolling(window=7).mean()
        df['price_ma30'] = df['price'].rolling(window=30).mean()
        
        # Add day of week and month features
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        
        return df.dropna()
