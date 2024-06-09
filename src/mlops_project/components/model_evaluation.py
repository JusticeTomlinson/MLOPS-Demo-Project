import os
from src.mlops_project import logger
import pandas as pd
import mlflow
import mlflow.sklearn
import numpy as np
from urllib.parse import urlparse
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
import dagshub
from pathlib import Path

from src.mlops_project.utils.model_architecture import NeuralNetwork
from src.mlops_project.utils.common import save_json

from src.mlops_project.entity.config_entity import ModelEvaluationConfig



class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config=config
    
    def eval_metrics(self, actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_squared_error(actual, pred)
        r2 = r2_score(actual, pred)

        return rmse, mae, r2
    
    def log_into_mlflow(self):
        # Load the test data

        test_data = pd.read_csv(self.config.test_data_path)
        test_data = test_data.dropna()

        # Load the PyTorch model (assuming it's saved with torch.save)
        model = NeuralNetwork(self.config.input_dim,
                               self.config.hidden1_dim, 
                               self.config.hidden2_dim, 
                               self.config.output_dim)
        self.config

        model.load_state_dict(torch.load(self.config.model_path))

        model.eval()  # Set the model to evaluation mode

        # Prepare the test inputs and labels
        test_x = torch.tensor(test_data.drop([self.config.target_column], axis=1).values, dtype=torch.float32)
        test_y = torch.tensor(test_data[[self.config.target_column]].values, dtype=torch.float32)
        dagshub.init(repo_name="MLOPS-Demo-Project", repo_owner="JusticeTomlinson")


        # Set up MLflow
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():
            # Make predictions
            with torch.no_grad():  # Disable gradient calculation
                predicted_qualities = model(test_x)

            # Convert predictions to match the expected format
            predicted_qualities = predicted_qualities.numpy().flatten()

            # Evaluate metrics
            (rmse, mae, r2) = self.eval_metrics(test_y.numpy(), predicted_qualities)

            # Save metrics locally
            scores = {"rmse": rmse, "mae": mae, "r2": r2}
            save_json(path=Path(self.config.metric_file_name), data=scores)

            # Log parameters and metrics
            mlflow.log_params(self.config.all_params)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2", r2)
            mlflow.log_metric("mae", mae)

            # Handle model logging and registration

            if tracking_url_type_store != "file":
                # mlflow.pytorch.log_model(model, "model", registered_model_name="HearnDiseaseNN")

                mlflow.pytorch.log_model(pytorch_model=model, 
                                         artifact_path="model",
                                         registered_model_name="NeuralNetwork")



            else:
                # mlflow.pytorch.log_model(model, "model")
                # mlflow.pytorch.log_model(model, "model", pip_requirements="pip_requirements.txt")
                mlflow.pytorch.log_model(pytorch_model=model, 
                                         artifact_path="model",
                                         registered_model_name="NeuralNetwork")

