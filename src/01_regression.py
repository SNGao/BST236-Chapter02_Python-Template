"""This module performs data loading, preprocessing, and regression analysis
on the Boston Housing dataset using a Random Forest Regressor.
"""

import os
import yaml
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import json
import config.config as config

class DataLoader:
    def __init__(self, config_path='src/config/config.yaml'):
        """Initialize DataLoader with configuration"""
        self.config = self._load_config(config_path)
        self.random_seed = self.config['base']['params']['random_seed']
        np.random.seed(self.random_seed)
        
    def _load_config(self, config_path):
        """Load configuration from YAML file"""
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    
    def load_data(self):
        data_path = self.config['data']['processed_data']
        """Load the dataset"""
        self.data = pd.read_csv(data_path)
        return self.data
    
    def split_data(self, test_size=0.2):
        """Split data into features and target, then into train and test sets"""
        self.X = self.data.drop('MEDV', axis=1)
        self.y = self.data['MEDV']
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, 
            test_size=test_size,
            random_state=self.random_seed
        )
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def get_model_params(self):
        """Get model parameters from config"""
        return {
            'n_estimators': self.config['model']['parameters']['n_estimators'],
            'max_depth': self.config['model']['parameters']['max_depth'],
            'random_state': self.random_seed
        }
    
    @staticmethod
    def calculate_metrics(y_true, y_pred, dataset_name):
        """Calculate and print regression metrics"""
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        print(f"\n{dataset_name} Metrics:")
        print(f"MSE: {mse:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"R2 Score: {r2:.2f}")
        
        return {'mse': mse, 'rmse': rmse, 'r2': r2}
    
    def get_feature_importance(self, model):
        """Calculate and return feature importance"""
        feature_importance = pd.DataFrame({
            'feature': self.X.columns,
            'importance': model.feature_importances_
        })
        return feature_importance.sort_values('importance', ascending=False)
    
    def save_results(self, metrics, feature_importance, output_dir='src/output'):
        """Save metrics and feature importance to files"""
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save metrics
        metrics_file = os.path.join(output_dir, 'metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        # Save feature importance
        importance_file = os.path.join(output_dir, 'feature_importance.csv')
        feature_importance.to_csv(importance_file, index=False)


if __name__ == "__main__":
    # Initialize DataLoader
    CONFIG_PATH = 'src/config/config.yaml'
    data_loader = DataLoader(config_path=CONFIG_PATH)
    
    # Load and split data
    data_loader.load_data()
    X_train, X_test, y_train, y_test = data_loader.split_data()
    
    # Initialize and train model
    model_params = data_loader.get_model_params()
    rf_model = RandomForestRegressor(**model_params)
    rf_model.fit(X_train, y_train)
    
    # Make predictions
    train_predictions = rf_model.predict(X_train)
    test_predictions = rf_model.predict(X_test)
    
    # Calculate metrics
    train_metrics = data_loader.calculate_metrics(y_train, train_predictions, "Training")
    test_metrics = data_loader.calculate_metrics(y_test, test_predictions, "Test")
    
    # Get feature importance
    feature_importance = data_loader.get_feature_importance(rf_model)
    print("\nTop 5 Most Important Features:")
    print(feature_importance.head())
    
    # Save results
    metrics = {
        'training': train_metrics,
        'test': test_metrics
    }
    data_loader.save_results(metrics, feature_importance)
