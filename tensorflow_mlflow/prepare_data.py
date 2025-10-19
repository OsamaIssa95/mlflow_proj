#basics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#machine learning libs
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

#deeplearning libs
import tensorflow as tf
from tensorflow import keras
#additional deep learning libs
from hyperopt import fmin, toe, hp, STATUS_OK, Trials

#mlops libs
import mlflow
from mlflow.models import infer_signature

#important libs for clean coding
from typing import Dict, Tuple, Any, Optional
import logging
import warnings


logger = logging.getLogger("__main__")

class DataProcessor:
    """ Handles data loading, preprocessing, and splitting"""
    def __init__(self, separator: str = ";" ,test_size: float = 0.25, val_size: float = 0.25, random_state: int = SEED):
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.separator = separator
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def load_data(self, url: str)->pd.DataFrame:
        """ Load dataset from URL"""
        logger.info(f"Loading data from {url}")
        try:
            data = pd.read_csv(url, sep=";")
            logger.info(f"Loaded dataset with shape: {data.shape}")
            return data
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
        
    def prepare_features_target(self, data: pd.DataFrame,
                                target_col: str = "quality")->Tuple[pd.DataFrame, pd.Series]:
        """"Separation features and target"""
        x = data.drop(columns =[target_col])
        y = data[target_col]
        return x,y
    
    def split_data(self, x: pd.DataFrame, y: pd.Series) ->Dict[str, np.ndarray]:
        """Create train/validation/test splits with proper scaling"""
        # Initial train/test split
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=self.test_size, random_state=self.random_state
        )
        
        # Further split training data for validation
        x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=self.val_size, random_state=self.random_state
        )
        
        # Scale feature
        if not self.is_fitted:
            x_train_scaled = self.scaler.fit_transform(x_train)
            x_val_scaled = self.scaler.fit_transform(x_val)
            x_test_scaled = self.scaler.fit_transform(x_test)
            self.is_fitted = True
        
        logger.info(f"Data splits - Train: {x_train_scaled.shape}, Val: {x_val_scaled.shape},Test: {x_test_scaled.shape}")
        return {
            'x_train': x_train_scaled, 'y_train': y_train.values,
            'x_val': x_val_scaled, 'y_val': y_val.values,
            'x_test': x_test_scaled, 'y_test': y_test.values
        }
        

        


