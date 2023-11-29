import logging

import mlflow
from zenml import step

from zenml.integrations.mlflow.steps.mlflow_registry import mlflow_register_model_step
from sklearn.base import ClassifierMixin


@step
def process_register_model(model: ClassifierMixin, name=str) -> None:
    pass
