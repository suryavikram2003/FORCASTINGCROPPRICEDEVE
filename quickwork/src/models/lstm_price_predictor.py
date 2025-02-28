import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Bidirectional, Attention, LayerNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import optuna

class AdvancedAgriculturePricePredictor:
    def __init__(self, sequence_length=60):
        self.sequence_length = sequence_length
        self.scaler = RobustScaler()
        self.model = None
        self.history = None
        self.n_features = None  # Will be set dynamically based on data
    
    def create_sequences(self, data, target_col=0):
        X, y = [], []
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:(i + self.sequence_length)])
            y.append(data[i + self.sequence_length, target_col])
        return np.array(X), np.array(y)
    
    def build_advanced_model(self, input_shape, lstm_units=128, dropout_rate=0.2):
        inputs = Input(shape=input_shape)
        
        # Bidirectional LSTM layers with return sequences
        x = Bidirectional(LSTM(lstm_units, return_sequences=True))(inputs)
        x = LayerNormalization()(x)
        x = Dropout(dropout_rate)(x)
        
        x = Bidirectional(LSTM(lstm_units//2, return_sequences=True))(x)
        x = LayerNormalization()(x)
        x = Dropout(dropout_rate)(x)
        
        # Attention mechanism
        attention = Attention()([x, x])
        
        # Final LSTM layer
        x = LSTM(lstm_units//4)(attention)
        x = LayerNormalization()(x)
        x = Dropout(dropout_rate)(x)
        
        # Dense layers with residual connection
        x1 = Dense(64, activation='relu')(x)
        x2 = Dense(32, activation='relu')(x1)
        outputs = Dense(1)(x2)
        
        model = Model(inputs=inputs, outputs=outputs)
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae', 'mape']
        )
        return model
    
    def optimize_hyperparameters(self, X, y, n_trials=20):
        def objective(trial):
            lstm_units = trial.suggest_int('lstm_units', 64, 256)
            dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
            batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
            
            model = self.build_advanced_model(
                (self.sequence_length, self.n_features),
                lstm_units=lstm_units,
                dropout_rate=dropout_rate
            )
            
            history = model.fit(
                X, y,
                epochs=10,
                batch_size=batch_size,
                validation_split=0.2,
                verbose=0
            )
            
            return min(history.history['val_loss'])
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        return study.best_params
    
    def train_model(self, data, commodity='Rice', epochs=100):
        # Data preprocessing
        commodity_data = data[data['commodity'] == commodity].sort_values('date')
        
        # Feature engineering
        commodity_data['price_change'] = commodity_data['price'].pct_change()
        commodity_data['price_ma7'] = commodity_data['price'].rolling(window=7).mean()
        commodity_data['price_volatility'] = commodity_data['price'].rolling(window=7).std()
        commodity_data = commodity_data.dropna()
        
        features = ['price', 'temperature', 'rainfall', 'demand', 'volume',
                   'market_sentiment', 'global_demand_index', 'price_change',
                   'price_ma7', 'price_volatility']
        
        self.n_features = len(features)  # Dynamically set number of features
        scaled_data = self.scaler.fit_transform(commodity_data[features])
        X, y = self.create_sequences(scaled_data)
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Optimize hyperparameters
        best_params = self.optimize_hyperparameters(X, y)
        
        self.model = self.build_advanced_model(
            (self.sequence_length, self.n_features),
            lstm_units=best_params['lstm_units'],
            dropout_rate=best_params['dropout_rate']
        )
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
            ModelCheckpoint(f'best_model_{commodity}.h5', monitor='val_loss', save_best_only=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
        ]
        
        # Training with cross-validation
        histories = []
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=best_params['batch_size'],
                callbacks=callbacks,
                verbose=1
            )
            histories.append(history)
        
        self.history = histories[-1]  # Store last fold's history
        return self.evaluate_model(X, y)
    
    def evaluate_model(self, X, y):
        predictions = self.model.predict(X)
        actual = y
        
        metrics = {
            'mae': mean_absolute_error(actual, predictions),
            'mape': np.mean(np.abs((actual - predictions) / actual)) * 100,
            'r2': r2_score(actual, predictions)
        }
        
        self.plot_metrics()
        return metrics
    
    def plot_metrics(self):
        plt.figure(figsize=(15, 10))
        
        # Loss plot
        plt.subplot(2, 2, 1)
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # MAE plot
        plt.subplot(2, 2, 2)
        plt.plot(self.history.history['mae'], label='Training MAE')
        plt.plot(self.history.history['val_mae'], label='Validation MAE')
        plt.title('Model MAE')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('advanced_training_metrics.png')
        plt.close()
    
    def predict_future(self, last_sequence, days=30):
        predictions = []
        current_sequence = last_sequence.copy()
        
        for _ in range(days):
            pred = self.model.predict(current_sequence.reshape(1, self.sequence_length, -1), verbose=0)
            predictions.append(pred[0, 0])
            
            # Update sequence with realistic feature values
            new_row = current_sequence[-1].copy()
            new_row[0] = pred[0, 0]  # Update price
            current_sequence = np.roll(current_sequence, -1, axis=0)
            current_sequence[-1] = new_row
        
        predictions = np.array(predictions).reshape(-1, 1)
        dummy_features = np.zeros((len(predictions), self.n_features - 1))  # Exclude price column
        full_predictions = np.hstack([predictions, dummy_features])
        predictions = self.scaler.inverse_transform(full_predictions)[:, 0]
        
        return predictions

def main():
    data = pd.read_csv('agricultural_prices_dataset.csv')
    data['date'] = pd.to_datetime(data['date'])
    
    predictor = AdvancedAgriculturePricePredictor()
    
    commodities = ['Rice', 'Wheat', 'Corn']
    results = {}
    
    for commodity in commodities:
        print(f"\nTraining model for {commodity}")
        metrics = predictor.train_model(data, commodity=commodity)
        results[commodity] = metrics
        
        # Prepare data for prediction
        commodity_data = data[data['commodity'] == commodity].sort_values('date')
        commodity_data['price_change'] = commodity_data['price'].pct_change()
        commodity_data['price_ma7'] = commodity_data['price'].rolling(window=7).mean()
        commodity_data['price_volatility'] = commodity_data['price'].rolling(window=7).std()
        commodity_data = commodity_data.dropna()
        
        features = ['price', 'temperature', 'rainfall', 'demand', 'volume',
                   'market_sentiment', 'global_demand_index', 'price_change',
                   'price_ma7', 'price_volatility']
        last_sequence = predictor.scaler.transform(commodity_data[features].iloc[-60:])
        predictions = predictor.predict_future(last_sequence)
        
        print(f"\n{commodity} - Next 30 days predictions:")
        for i, price in enumerate(predictions, 1):
            print(f"Day {i}: Rs.{price:.2f}")
    
    print("\nTraining Results Summary:")
    for commodity, metrics in results.items():
        print(f"\n{commodity}:")
        print(f"MAE: ₹{metrics['mae']:.2f}")
        print(f"MAPE: {metrics['mape']:.2f}%")
        print(f"R2 Score: {metrics['r2']:.4f}")

if __name__ == "__main__":
    main()