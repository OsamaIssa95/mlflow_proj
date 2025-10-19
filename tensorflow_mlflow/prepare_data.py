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
    def __init__(self, seperator: str = ";",test_size: float = 0.25, val_size: float = 0.25, random_state: int = SEED):
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.seperator = seperator
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

        


