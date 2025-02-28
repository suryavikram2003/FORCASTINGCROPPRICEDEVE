from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

class ModelEvaluator:
    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred

    def calculate_metrics(self):
        return {
            'mse': mean_squared_error(self.y_true, self.y_pred),
            'rmse': mean_squared_error(self.y_true, self.y_pred, squared=False),
            'mae': mean_absolute_error(self.y_true, self.y_pred),
            'r2': r2_score(self.y_true, self.y_pred)
        }

    def plot_predictions(self, save_path=None):
        plt.figure(figsize=(12, 6))
        plt.plot(self.y_true, label='Actual')
        plt.plot(self.y_pred, label='Predicted')
        plt.title('Price Prediction vs Actual')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        
        if save_path:
            plt.savefig(save_path)
        plt.close()
