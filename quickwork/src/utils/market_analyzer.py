from flask import Flask, jsonify, request
from flask_cors import CORS
import requests
from datetime import datetime, timedelta

app = Flask(__name__)
CORS(app)

class MarketAPI:
    BASE_URL = "https://api.data.gov.in/resource/35985678-0d79-46b4-9ed6-6f13308a1d24"
    API_KEY = "579b464db66ec23bdd000001b4f219089a674ef26e56317cad12233c"

    def get_market_data(self):
        # Try to get data for the last 7 days until we find records
        for days_back in range(7):
            check_date = datetime.now() - timedelta(days=days_back)
            formatted_date = check_date.strftime('%d/%m/%Y')
            
            data = self.fetch_data_for_date(formatted_date)
            if data.get('records'):
                print(f"Found recent data for {formatted_date}")
                return data
                
        # If no data found in last 7 days, get latest available data
        params = {
            'api-key': self.API_KEY,
            'format': 'json',
            'filters[State.keyword]': 'Tamil Nadu',
            'filters[District.keyword]': 'Salem',
            'sort[Arrival_Date]': 'desc',  # Sort by date descending
            'limit': 100
        }
        
        return self.make_api_request(params)

    def fetch_data_for_date(self, date):
        params = {
            'api-key': self.API_KEY,
            'format': 'json',
            'filters[State.keyword]': 'Tamil Nadu',
            'filters[District.keyword]': 'Salem',
            'filters[Arrival_Date]': date,
            'limit': 100
        }
        
        return self.make_api_request(params)

    def make_api_request(self, params):
        try:
            response = requests.get(self.BASE_URL, params=params)
            response.raise_for_status()
            data = response.json()
            
            if data and 'records' in data:
                transformed_records = [{
                    'state': record['State'],
                    'district': record['District'],
                    'market': record['Market'],
                    'commodity': record['Commodity'],
                    'variety': record['Variety'],
                    'arrival_date': record['Arrival_Date'],
                    'min_price': float(record['Min_Price']),
                    'max_price': float(record['Max_Price']),
                    'modal_price': float(record['Modal_Price'])
                } for record in data['records']]
                
                data['records'] = transformed_records
                print(f"Successfully fetched {len(data['records'])} records")
            
            return data
            
        except Exception as e:
            print(f"Data fetch status: {str(e)}")
            return {'records': [], 'total': 0}

@app.route('/api/market-data')
def get_market_data():
    api = MarketAPI()
    data = api.get_market_data()
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True)
