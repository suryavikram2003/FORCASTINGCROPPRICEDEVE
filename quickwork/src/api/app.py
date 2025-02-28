from flask import Flask, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

app = Flask(__name__)
CORS(app)

class PricePredictor:
    def __init__(self, sequence_length=60):
        self.sequence_length = sequence_length
        self.model = self.build_model()
        
    def build_model(self):
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=(self.sequence_length, 7)),
            Dropout(0.2),
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            LSTM(32),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model
        
    def prepare_sequences(self, data):
        X, y = [], []
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:(i + self.sequence_length)])
            y.append(data[i + self.sequence_length, 0])
        return np.array(X), np.array(y)
        
    def train(self, data, epochs=50):
        X, y = self.prepare_sequences(data)
        self.model.fit(X, y, epochs=epochs, batch_size=32, verbose=1)
        
    def predict(self, sequence):
        return self.model.predict(sequence.reshape(1, self.sequence_length, -1))

@app.route('/api/predictions', methods=['GET'])
def get_predictions():
    data = pd.read_csv('D:/QuickWork/agricultural_prices_dataset.csv')

    predictions = {}
    
    for commodity in ['Rice', 'Wheat', 'Corn']:
        predictor = PricePredictor()
        commodity_data = data[data['commodity'] == commodity]
        features = ['price', 'temperature', 'rainfall', 'demand', 
                   'volume', 'market_sentiment', 'global_demand_index']
        
        # Prepare and scale data
        feature_data = commodity_data[features].values
        predictor.train(feature_data)
        
        # Make predictions
        last_sequence = feature_data[-60:]
        future_prices = []
        
        for _ in range(30):
            next_price = predictor.predict(last_sequence)[0][0]
            future_prices.append(float(next_price))
            last_sequence = np.roll(last_sequence, -1, axis=0)
            last_sequence[-1] = np.array([next_price] + list(last_sequence[-1][1:]))
            
        predictions[commodity] = {
            'prices': future_prices,
            'dates': [(pd.Timestamp.now() + pd.Timedelta(days=i)).strftime('%Y-%m-%d') 
                     for i in range(30)]
        }
    
    return jsonify(predictions)

if __name__ == '__main__':
    app.run(debug=True)
