from flask_socketio import SocketIO, emit
import threading
import time
import random
from datetime import datetime

class PriceHandler:
    def __init__(self, app):
        self.socketio = SocketIO(app, cors_allowed_origins="*")
        self.commodities = {
            'Rice': {'base_price': 45.0},
            'Wheat': {'base_price': 32.0},
            'Corn': {'base_price': 28.0},
            'Soybean': {'base_price': 38.0}
        }
        self.setup_socket_events()
        self.start_price_updates()

    def setup_socket_events(self):
        @self.socketio.on('connect', namespace='/prices')
        def handle_connect():
            emit('price_update', self.get_current_prices())

        @self.socketio.on('refresh', namespace='/prices')
        def handle_refresh():
            emit('price_update', self.get_current_prices())

    def get_current_prices(self):
        current_prices = {}
        for commodity, data in self.commodities.items():
            base_price = data['base_price']
            variation = random.uniform(-0.5, 0.5)
            current_price = base_price + variation
            
            current_prices[commodity] = {
                'price': round(current_price, 2),
                'trend': 'up' if variation > 0 else 'down',
                'change': round(variation/base_price * 100, 2),
                'volume': random.randint(1000, 5000),
                'confidence': random.randint(85, 98)
            }
        return current_prices

    def start_price_updates(self):
        def price_updater():
            while True:
                self.socketio.emit(
                    'price_update', 
                    self.get_current_prices(), 
                    namespace='/prices'
                )
                time.sleep(5)  # Update every 5 seconds

        thread = threading.Thread(target=price_updater)
        thread.daemon = True
        thread.start()
