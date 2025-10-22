#basics
from gc import callbacks
from tabnanny import verbose
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

logger = logging.getLogger("__main__")


class WineQualityModel:
    """Neural network model for wine qualiy predection"""
    
    def __init__(self, input_dim: int):
        self.input_dim = input_dim
        self.model = None
        self.history = None
        
    def build_model(self, learning_rate: float, momentum: float, hidden_layers: Tuple[int, ...] = (64, 32), dropout_rate: float = 0.2) -> keras.model:
        """"Build and compile the neural network model"""
        model = keras.Sequential()
        model.add(keras.layers.Input(shape=(self.input_dim,)))
        
        # Add hidden layers
        for units in hidden_layers:
            model.add(keras.layers.Dense(units, activation='relu'))
            if dropout_rate > 0:
                model.add(keras.layers.Dropout(dropout_rate))
                
        #Output layer
        model.add(keras.layers.Dense(1))
        
        #Compile model
        optimizer = keras.optimizer.adam(
            learning_rate=learning_rate,
            momentum=momentum
        )
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=[keras.metrics.RootMeanSquaredError(), keras.metrics.MeanAbsoluteError()]
        )
        self.model = model
        return model
    
    def train(self, x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray, epochs: int = 50, batch_size: int = 32, patience: int = 10) -> Dict[str, Any]:
        """Train the Data with early stopping"""
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_los',
            patience=patience,
            restore_best_weights=True,
            verbose=0
        )
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=0
        )
        self.history = self.model.fit(
            x_train, y_train, validation_data=(x_val, y_val), epochs=epochs, callbacks=[early_stopping, reduce_lr], verbose=0
        )
        # Evaluate on validation set
        val_loss, val_rms, val_mae = self.model.evaluate(x_val, y_val, verbose=0)
        
        return {
            "val_loss": val_loss,
            "val_rms": val_rms,
            "val_mae": val_mae,
            "epochs_trained": len(self.history.history["loss"]),
            "history": self.history
        }