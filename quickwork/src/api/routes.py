from flask import Blueprint, request, jsonify
from models.price_predictor import PricePredictor
from utils.market_analyzer import MarketAnalyzer
from datetime import datetime, timedelta

api = Blueprint('api', __name__)
predictor = PricePredictor()
analyzer = MarketAnalyzer()

@api.route('/prices', methods=['GET'])
def get_prices():
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        historical_data = analyzer.get_historical_prices(start_date, end_date)
        predictions = predictor.get_future_predictions(days=7)
        
        return jsonify({
            'historical': historical_data,
            'predictions': predictions
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api.route('/predict', methods=['POST'])
def predict_price():
    try:
        data = request.get_json()
        commodity = data['commodity']
        quantity = float(data['quantity'])
        
        prediction = predictor.predict_price(commodity, quantity)
        market_analysis = analyzer.get_market_sentiment(commodity)
        
        return jsonify({
            'price': prediction,
            'market_analysis': market_analysis
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
