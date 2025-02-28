from flask import Blueprint, jsonify
from models.lstm_price_predictor import AgriculturePricePredictor
import pandas as pd

prediction_api = Blueprint('prediction_api', __name__)

@prediction_api.route('/predictions', methods=['GET'])
def get_predictions():
    data = pd.read_csv('agricultural_prices_dataset.csv')
    predictor = AgriculturePricePredictor()
    
    predictions = {}
    commodities = ['Rice', 'Wheat', 'Corn']
    
    for commodity in commodities:
        commodity_data = data[data['commodity'] == commodity].sort_values('date')
        features = ['price', 'temperature', 'rainfall', 'demand', 
                   'volume', 'market_sentiment', 'global_demand_index']
        last_sequence = commodity_data[features].iloc[-60:].values
        future_prices = predictor.predict_future(last_sequence)
        
        predictions[commodity] = [
            {
                'day': i + 1,
                'price': float(price),
                'date': (pd.Timestamp.now() + pd.Timedelta(days=i)).strftime('%Y-%m-%d')
            }
            for i, price in enumerate(future_prices)
        ]
    
    return jsonify(predictions)
