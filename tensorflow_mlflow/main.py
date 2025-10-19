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

#configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')




def main():
    pass





if __name__ == "__main__":
    main()