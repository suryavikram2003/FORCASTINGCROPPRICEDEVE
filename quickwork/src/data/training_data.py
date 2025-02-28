import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def create_training_dataset():
    # Create 3 years of daily data
    dates = pd.date_range(start='2023-01-01', end='2025-01-01', freq='D')
    
    # Base prices for different commodities
    commodities = {
        'Rice': {
            'base_price': 45.0,
            'seasonal_factor': 0.15,
            'weather_sensitivity': 0.2,
            'demand_sensitivity': 0.3
        },
        'Wheat': {
            'base_price': 32.0,
            'seasonal_factor': 0.12,
            'weather_sensitivity': 0.25,
            'demand_sensitivity': 0.28
        },
        'Corn': {
            'base_price': 28.0,
            'seasonal_factor': 0.18,
            'weather_sensitivity': 0.22,
            'demand_sensitivity': 0.25
        }
    }
    
    all_data = []
    
    for date in dates:
        for commodity, params in commodities.items():
            # Seasonal component
            day_of_year = date.dayofyear
            seasonal = np.sin(2 * np.pi * day_of_year / 365) * params['seasonal_factor']
            
            # Weather component
            temperature = np.random.normal(25, 5)  # Mean temp 25°C
            rainfall = max(0, np.random.normal(50, 20))  # Mean rainfall 50mm
            weather_effect = (temperature - 25) * 0.01 + (rainfall - 50) * 0.005
            
            # Market demand component
            base_demand = 1000 + np.random.normal(0, 100)
            demand_multiplier = base_demand / 1000
            
            # Calculate final price
            base_price = params['base_price']
            weather_impact = weather_effect * params['weather_sensitivity']
            demand_impact = (demand_multiplier - 1) * params['demand_sensitivity']
            
            price = base_price * (1 + seasonal + weather_impact + demand_impact)
            
            # Add random market noise
            price *= (1 + np.random.normal(0, 0.02))
            
            all_data.append({
                'date': date,
                'commodity': commodity,
                'price': round(max(price, base_price * 0.7), 2),  # Ensure price doesn't go too low
                'temperature': round(temperature, 1),
                'rainfall': round(rainfall, 1),
                'demand': round(base_demand),
                'volume': round(base_demand * np.random.uniform(0.8, 1.2)),
                'month': date.month,
                'day_of_week': date.dayofweek,
                'season': (date.month % 12 + 3) // 3,
                'market_sentiment': round(np.random.uniform(-1, 1), 2),
                'global_demand_index': round(np.random.normal(100, 10), 2),
                'export_quota': round(np.random.uniform(90, 110), 2)
            })
    
    df = pd.DataFrame(all_data)
    
    # Add trends and patterns
    df['moving_avg_7d'] = df.groupby('commodity')['price'].transform(lambda x: x.rolling(7).mean())
    df['price_momentum'] = df.groupby('commodity')['price'].transform(lambda x: x.pct_change())
    
    # Save the dataset
    df.to_csv('agricultural_prices_dataset.csv', index=False)
    print(f"Generated {len(df)} records of training data")
    
    return df

def load_training_data():
    try:
        return pd.read_csv('agricultural_prices_dataset.csv', parse_dates=['date'])
    except:
        return create_training_dataset()

if __name__ == "__main__":
    dataset = create_training_dataset()
    print("\nSample Data Preview:")
    print(dataset.head())
    print("\nDataset Statistics:")
    print(dataset.describe())
    
    # Display unique values in categorical columns
    categorical_cols = ['commodity', 'month', 'day_of_week', 'season']
    for col in categorical_cols:
        print(f"\nUnique values in {col}:")
        print(dataset[col].value_counts())
