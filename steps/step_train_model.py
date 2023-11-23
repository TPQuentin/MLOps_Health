import logging
import mlflow
import pandas as pd
from components.component_train_model import ModelTraining
from sklearn.base import ClassifierMixin

from zenml.client import Client
from zenml.steps import step
from .config import ModelNameConfig

# set experiment_tracker variable with the expriement tracker active in the current active stack
experiment_tracker = Client().active_stack.experiment_tracker


@step(experiment_tracker=experiment_tracker.name)
def process_train_model(X_train: pd.DataFrame, X_test: pd.DataFrame,
                        y_train: pd.Series, y_test: pd.Series, config: ModelNameConfig) -> ClassifierMixin:
    """
    Args:
        X_train: The train data
        X_test: The test data
        y_train: The target for train data
        y_test: The target for test data

    Returns:
        model: artefact representing the trained model
    """
    try:
        model_training = ModelTraining(X_train, y_train, X_test, y_test)

        if config.model_name == "lightgbm":
            mlflow.lightgbm.autolog()
            lgm_model = model_training.lightgbm_trainer(
                fine_tuning=config.fine_tuning
            )
            return lgm_model
        elif config.model_name == "randomforest":
            mlflow.sklearn.autolog()
            rf_model = model_training.random_forest_trainer(
                fine_tuning=config.fine_tuning
            )
            return rf_model
        elif config.model_name == "xgboost":
            mlflow.xgboost.autolog()
            xgb_model = model_training.xgboost_trainer(
                fine_tuning=config.fine_tuning
            )
            return xgb_model
        else:
            raise ValueError("Model name not supported")
    except Exception as e:
        logging.error(e)
        raise e
