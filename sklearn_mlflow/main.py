from mlflow import MlflowClient
from pprint import pprint
from mlflow.entities import experiment
from training_code import data_val,prepare_data,train_model,params
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


def safe_experiment_setup(experiment_name=None):
    """
    Safely set up MLflow experiment, handling various edge cases
    """
    if experiment_name is None:
        experiment_name = os.getenv("MLFLOW-EXPERIMENT-NAME", "Default_Experiment")
    
    client = MlflowClient()
    
    try:
        # Try to get the experiment
        experiment = client.get_experiment_by_name(experiment_name)
        
        if experiment:
            if experiment.lifecycle_stage == "deleted":
                # Restore if deleted
                client.restore_experiment(experiment.experiment_id)
                print(f"✅ Restored previously deleted experiment: {experiment_name}")
            else:
                print(f"✅ Using existing experiment: {experiment_name}")
        else:
            # Create new experiment if it doesn't exist
            experiment_id = mlflow.create_experiment(experiment_name)
            print(f"✅ Created new experiment: {experiment_name}")
        
        # Set as active experiment
        return mlflow.set_experiment(experiment_name)
        
    except Exception as e:
        print(f"❌ Error setting up experiment: {e}")
        # Fallback: create with modified name
        fallback_name = f"{experiment_name}_fallback"
        print(f"🔄 Trying fallback experiment: {fallback_name}")
        return mlflow.create_experiment(fallback_name)


def get_artifact_internal_id(run_id, experiment_id=None):
    """Extract the internal artifact ID from the filesystem"""
    
    if experiment_id is None:
        experiment_id = mlflow.active_run().info.experiment_id
    
    artifacts_dir = f"mlruns/{experiment_id}/{run_id}/outputs"
    
    if os.path.exists(artifacts_dir):
        for item in os.listdir(artifacts_dir):
            if item.startswith('m-'): 
                return item


#usage
apple_experiment = safe_experiment_setup()
run_name = os.getenv("MLFLOW-RUN-NAME")
artifact_path = os.getenv("MLFLOW-ARTIFACT-PATH", "model")
data_path = "./data/apple_sales_data.csv"   


def main():
    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        print(f"{e},generating data...")
        df = generate_apple_sales_data_with_promo_adjustment()
        df.to_csv(data_path)
        print("Data was generated succesfully")
    x_train, x_val, y_train, y_val = prepare_data(df)
    rf = train_model(x_train=x_train, y_train=y_train, params=params)
    metrics = data_val(model=rf, x_val=x_val, y_val=y_val)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        model_path = "model"
        mlflow.sklearn.log_model(sk_model=rf, input_example=x_val, registered_model_name="apples_model", artifact_path=model_path)
        #mlflow.log_artifact(data_path, artifact_path="data")
        """ mlflow.register_model(
            model_uri=f"runs:/{run.info.run_id}/{model_path}",
            name="apples_model"
        ) """
          # Hard code part, cause the log_model is not working properly
        dest_id = get_artifact_internal_id(run.info._run_id)
        mlflow.log_artifacts(f"./mlartifacts/{run.info.experiment_id}/models/{dest_id}/artifacts", artifact_path=artifact_path)


if __name__ == "__main__":
    main()