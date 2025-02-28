import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_commodity_dataset():
    # Base dataset with daily prices for multiple commodities
    start_date = datetime(2022, 1, 1)
    dates = pd.date_range(start=start_date, end=datetime.now(), freq='D')
    
    commodities_data = {
        'Rice': {
            'base_price': 45.0,
            'seasonality': 0.15,
            'volatility': 0.08
        },
        'Wheat': {
            'base_price': 32.0,
            'seasonality': 0.12,
            'volatility': 0.10
        },
        'Corn': {
            'base_price': 28.0,
            'seasonality': 0.18,
            'volatility': 0.09
        },
        'Soybean': {
            'base_price': 38.0,
            'seasonality': 0.14,
            'volatility': 0.11
        }
    }

    all_data = []
    
    for date in dates:
        day_of_year = date.dayofyear
        for commodity, params in commodities_data.items():
            # Add seasonal effect
            seasonal_factor = np.sin(2 * np.pi * day_of_year / 365) * params['seasonality']
            
            # Add random volatility
            volatility = np.random.normal(0, params['volatility'])
            
            # Calculate price with trends and patterns
            price = params['base_price'] * (1 + seasonal_factor + volatility)
            
            # Generate related features
            volume = np.random.randint(1000, 5000)
            rainfall = np.random.uniform(0, 100) # mm
            temperature = np.random.uniform(20, 35) # Celsius
            humidity = np.random.uniform(40, 90) # percentage
            
            all_data.append({
                'date': date,
                'commodity': commodity,
                'price': round(price, 2),
                'volume': volume,
                'rainfall': round(rainfall, 2),
                'temperature': round(temperature, 2),
                'humidity': round(humidity, 2),
                'month': date.month,
                'day_of_week': date.dayofweek,
                'season': (date.month % 12 + 3) // 3
            })
    
    return pd.DataFrame(all_data)

def generate_market_factors():
    # Additional market influence factors
    dates = pd.date_range(start='2025-01-01', end=datetime.now(), freq='D')
    
    market_data = []
    
    for date in dates:
        market_data.append({
            'date': date,
            'global_demand_index': round(np.random.uniform(80, 120), 2),
            'export_quota': round(np.random.uniform(90, 110), 2),
            'market_sentiment': round(np.random.uniform(-1, 1), 2),
            'storage_capacity': round(np.random.uniform(70, 100), 2),
            'transportation_cost': round(np.random.uniform(95, 105), 2)
        })
    
    return pd.DataFrame(market_data)

# Save datasets
def save_sample_datasets():
    commodity_df = generate_commodity_dataset()
    market_df = generate_market_factors()
    
    # Save to CSV
    commodity_df.to_csv('D:/QuickWork/quickwork/src/data/commodity_prices.csv', index=False)
    market_df.to_csv('D:/QuickWork/quickwork/src/data/market_factors.csv', index=False)
    
    # Save to JSON for API testing
    commodity_df.to_json('D:/QuickWork/quickwork/src/data/commodity_prices.json', orient='records', date_format='iso')
    market_df.to_json('D:/QuickWork/quickwork/src/data/market_factors.json', orient='records', date_format='iso')
    
    return {
        'commodity_data': commodity_df,
        'market_data': market_df
    }

if __name__ == "__main__":
    datasets = save_sample_datasets()
    print(f"Generated {len(datasets['commodity_data'])} commodity records")
    print(f"Generated {len(datasets['market_data'])} market factor records")
