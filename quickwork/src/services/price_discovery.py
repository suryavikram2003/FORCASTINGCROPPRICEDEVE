import requests
import pandas as pd
from datetime import datetime
from typing import Dict, List

class AgriPriceService:
    def __init__(self):
        self.api_key = "579b464db66ec23bdd0000011c66a6e109084bad6d2eb0cb5aa45398"
        self.base_url = "https://api.data.gov.in/resource"
        self.resource_id = "9ef84268-d588-465a-a308-a864a43d0070"
    
    def get_market_prices(self, filters: Dict = None) -> Dict:
        params = {
            'api-key': self.api_key,
            'format': 'json',
            'limit': 100,
            'offset': 0
        }
        
        if filters:
            params.update(filters)
            
        url = f"{self.base_url}/{self.resource_id}"
        response = requests.get(url, params=params)
        return response.json()
    
    def get_commodity_prices(self, commodity: str) -> List[Dict]:
        params = {
            'filters[commodity]': commodity
        }
        data = self.get_market_prices(params)
        return data.get('records', [])
    
    def get_state_prices(self, state: str) -> List[Dict]:
        params = {
            'filters[state]': state
        }
        data = self.get_market_prices(params)
        return data.get('records', [])

class PriceAnalytics:
    def analyze_price_trends(self, records: List[Dict]) -> Dict:
        df = pd.DataFrame(records)
        analysis = {
            'average_price': df['modal_price'].mean(),
            'price_range': {
                'min': df['min_price'].min(),
                'max': df['max_price'].max()
            },
            'price_by_market': df.groupby('market')['modal_price'].mean().to_dict(),
            'total_markets': df['market'].nunique(),
            'updated_date': datetime.now().isoformat()
        }
        return analysis

# Flask Blueprint for API endpoints
from flask import Blueprint, jsonify, request

price_api = Blueprint('price_api', __name__)

@price_api.route('/prices/commodity/<commodity>', methods=['GET'])
def get_commodity_prices(commodity):
    service = AgriPriceService()
    analytics = PriceAnalytics()
    
    prices = service.get_commodity_prices(commodity)
    analysis = analytics.analyze_price_trends(prices)
    
    return jsonify({
        'commodity': commodity,
        'prices': prices,
        'analysis': analysis
    })

@price_api.route('/prices/state/<state>', methods=['GET'])
def get_state_prices(state):
    service = AgriPriceService()
    analytics = PriceAnalytics()
    
    prices = service.get_state_prices(state)
    analysis = analytics.analyze_price_trends(prices)
    
    return jsonify({
        'state': state,
        'prices': prices,
        'analysis': analysis
    })

@price_api.route('/prices/market', methods=['GET'])
def get_market_prices():
    market = request.args.get('market')
    commodity = request.args.get('commodity')
    
    service = AgriPriceService()
    params = {
        'filters[market]': market,
        'filters[commodity]': commodity
    }
    
    prices = service.get_market_prices(params)
    return jsonify(prices)
