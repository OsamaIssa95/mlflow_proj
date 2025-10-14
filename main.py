from mlflow import MlflowClient
from pprint import pprint
from mlflow.entities import experiment
from sklearn.ensemble import RandomForestRegressor
import dotenv
import os

dotenv.load_dotenv()
client = MlflowClient(tracking_uri=os.getenv("MLFLOW-TRACKING-URI"))
all_experiments = client.search_experiments()
default_experiment = [
    {"name": experiment.name, "lifecycle_stage": experiment.lifecycle_stage}
    for experiment in all_experiments
    if experiment.name == "Default"
][0]
pprint(default_experiment)