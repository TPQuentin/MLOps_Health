import logging

import mlflow
import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from zenml.client import Client
from zenml.steps import step
from typing import Tuple, Annotated

# Assuming this class is updated for classification
from components.component_evaluation import Evaluation

experiment_tracker = Client().active_stack.experiment_tracker


@step(experiment_tracker=experiment_tracker.name)
def process_evaluate(
    model: ClassifierMixin, X_test: pd.DataFrame, y_test: pd.Series
) -> Tuple[Annotated[float, "Accuracy"], Annotated[float, "Precision"],
           Annotated[float, "Recall"], Annotated[float, "F1_Score"]]:
    """
    Evaluates the performance of a classification model.

    Args:
        model: ClassifierMixin - A scikit-learn compatible classification model.
        X_test: pd.DataFrame - The test features.
        y_test: pd.Series - The true labels for the test set.

    Returns:
        accuracy: float - The accuracy of the model.
        precision: float - The precision of the model.
        recall: float - The recall of the model.
        f1_score: float - The F1 score of the model.
    """
    try:
        prediction = model.predict(X_test)
        evaluation = Evaluation()

        accuracy = evaluation.accuracy(y_test, prediction)
        mlflow.log_metric("accuracy", accuracy)

        precision = evaluation.precision(y_test, prediction, average='macro')
        mlflow.log_metric("precision", precision)

        recall = evaluation.recall(y_test, prediction, average='macro')
        mlflow.log_metric("recall", recall)

        f1 = evaluation.f1(y_test, prediction, average='macro')
        mlflow.log_metric("f1_score", f1)

        return accuracy, precision, recall, f1
    except Exception as e:
        logging.error(e)
        raise e
