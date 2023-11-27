import logging
from zenml import pipeline
from zenml.integrations.constants import MLFLOW
from zenml.config import DockerSettings
import os

from steps.step_ingest_data import process_ingest_data
from steps.step_clean_data import process_clean_data
from steps.step_train_model import process_train_model
from steps.step_evaluation import process_evaluate

docker_settings = DockerSettings(required_integrations=[MLFLOW])

FILE_NAME = os.path.basename(__file__)


@pipeline(enable_cache=False, settings={"docker": docker_settings})
def train_pipeline_test(path: str):
    """
    #process_ingest_data, process_clean_data, process_train_model, process_evaluate
    Args:
        step_ingest_data: DataClass
        step_clean_data: DataClass
        step_model_train: DataClass
        step_evaluation: DataClass
    Returns:
        mse: float
        rmse: float
    """
    logging.info(f"Starting {FILE_NAME}")
    df = process_ingest_data(path)

    x_train, x_test, y_train, y_test = process_clean_data(df)

    model = process_train_model(x_train, x_test, y_train, y_test)

    process_evaluate(model, x_test, y_test)
    logging.info(f"Ending {FILE_NAME}")
