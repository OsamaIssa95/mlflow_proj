# basics
from platform import processor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# machine learning libs
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# deeplearning libs
import tensorflow as tf
from tensorflow import keras
# additional deep learning libs
from hyperopt import fmin, toe, hp, STATUS_OK, Trials

# mlops libs
import mlflow
from mlflow.models import infer_signature

# important libs for clean coding
from typing import Dict, Tuple, Any, Optional
import logging
import warnings

import prepare_data
import model

# configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')




def main():
    """Main excution function"""
    
    # Configuration
    DATA_URL = "https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-white.csv"
    EXPEREMENT_NAME ="wine-quality-optimization-refactored"
    MAX_EVALS = 15
    
    try:
        # Step 1:Prepare data
        logger.info("Step 1: Prepare data")
        processor = prepare_data.DataProcessor()
        data = processor.load_data(DATA_URL)
        x, y = processor.prepare_features_target(data)
        processed_data = processor.split_data(x, y)
        
        # Step 2: Run hyperparameters optimization
        logger.info("Step 2: Starting hyperparameter optimnization")
        optimizer = model.HyperparameterOptimizer(processed_data, EXPEREMENT_NAME)
        results = optimizer.optimize(max_evals = MAX_EVALS)
        
        # Step 3: Final evaluation (optional - train best model on full training data)
        logger.info("Step 3: Optimization completed")
        logger.info(f"Best parameters: {results['best_params']}")
        logger.info(f"Best validation RMSE: {results['best_rmse']:.4f}")
        
    except Exception as e:
        logger.info(f"Pipline failed: {e}")
        raise
    

if __name__ == "__main__":
    main()