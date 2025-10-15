from mlflow import MlflowClient
from pprint import pprint
from mlflow.entities import experiment
from data_generator import generate_apple_sales_data_with_promo_adjustment
import mlflow
import pandas as pd
import dotenv
import os

dotenv.load_dotenv()
#using client api
""" client = MlflowClient(tracking_uri=os.getenv("MLFLOW-TRACKING-URI"))
all_experiments = client.search_experiments()
default_experiment = [
    {"name": experiment.name, "lifecycle_stage": experiment.lifecycle_stage}
    for experiment in all_experiments
    if experiment.name == "Default"
][0]
pprint(default_experiment) """
#using fluent api
mlflow.set_tracking_uri(os.getenv("MLFLOW-TRACKING-URI"))
apple_experiment = mlflow.set_experiment(os.getenv("MLFLOW-EXPERIMENT-NAME"))
run_name = os.getenv("MLFLOW-RUN-NAME")
artifact_path = os.getenv("MLFLOW-ARTIFACT-PATH")

data_path = "./data/apple_sales_data.csv"   
try:
    df = pd.read_csv(data_path)
except Exception as e:
    print(f"{e},generating data...")
    df = generate_apple_sales_data_with_promo_adjustment()
    df.to_csv(data_path)
    print("Data was generated succesfully")
    