from flask import Blueprint, request, jsonify
from models.lstm_model import PricePredictionModel
from utils.data_handler import DataHandler
import threading
import schedule
import time

price_api = Blueprint('price_api', __name__)
model = PricePredictionModel()
data_handler = DataHandler()

def train_models():
    for commodity in data_handler.commodities:
        historical_data = data_handler.generate_synthetic_data(commodity)
        model.train(historical_data)

@price_api.route('/predict', methods=['POST'])
def predict_price():
    data = request.get_json()
    commodity = data['commodity']
    
    historical_data = data_handler.generate_synthetic_data(commodity, days=30)
    market_factors = data_handler.get_market_factors()
    
    prediction = model.predict(historical_data)
    
    return jsonify({
        'commodity': commodity,
        'predicted_price': float(prediction),
        'confidence': 0.95,
        'market_factors': market_factors
    })

@price_api.route('/historical', methods=['GET'])
def get_historical():
    commodity = request.args.get('commodity', 'Rice')
    days = int(request.args.get('days', 30))
    
    data = data_handler.generate_synthetic_data(commodity, days)
    
    return jsonify({
        'commodity': commodity,
        'data': data.to_dict(orient='records')
    })

def schedule_training():
    schedule.every(24).hours.do(train_models)
    while True:
        schedule.run_pending()
        time.sleep(3600)

# Start the training schedule in a background thread
training_thread = threading.Thread(target=schedule_training)
training_thread.daemon = True
training_thread.start()

# Initial training
train_models()
