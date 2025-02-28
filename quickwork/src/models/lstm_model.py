import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

class PricePredictionModel:
    def __init__(self):
        self.model = None
        self.scaler = MinMaxScaler()
        self.sequence_length = 10
        self.features = ['price', 'volume', 'supply', 'demand']
        
    def create_model(self, input_shape):
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(64, return_sequences=False),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model

    def prepare_data(self, data):
        X, y = [], []
        scaled_data = self.scaler.fit_transform(data)
        
        for i in range(len(scaled_data) - self.sequence_length):
            X.append(scaled_data[i:(i + self.sequence_length)])
            y.append(scaled_data[i + self.sequence_length, 0])
            
        return np.array(X), np.array(y)

    def train(self, historical_data):
        X, y = self.prepare_data(historical_data[self.features])
        self.model = self.create_model((self.sequence_length, len(self.features)))
        self.model.fit(X, y, epochs=50, batch_size=32, validation_split=0.2)

    def predict(self, input_data):
        scaled_input = self.scaler.transform(input_data[self.features].values)
        X = scaled_input[-self.sequence_length:].reshape(1, self.sequence_length, len(self.features))
        scaled_prediction = self.model.predict(X)
        return self.scaler.inverse_transform(scaled_prediction)[0, 0]
