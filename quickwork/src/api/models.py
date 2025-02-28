from datetime import datetime

class MarketData:
    def __init__(self, db):
        self.collection = db['market_data']
    
    def insert_data(self, data):
        data['timestamp'] = datetime.now()
        return self.collection.insert_one(data)
    
    def get_all_data(self):
        return list(self.collection.find({}, {'_id': 0}))
    
    def update_data(self, query, data):
        data['timestamp'] = datetime.now()
        return self.collection.update_one(query, {'$set': data}, upsert=True)
