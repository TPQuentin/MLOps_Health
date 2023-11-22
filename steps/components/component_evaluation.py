import logging
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.base import ClassifierMixin
import pandas as pd

class Evaluation:
    """
    Evaluation class which evaluates the model performance for multiclass classification
    using sklearn metrics.
    """

    def __init__(self, y_true: np.ndarray, y_pred: np.ndarray, model: ClassifierMixin) -> None:
        """Initializes the Evaluation class."""
        self.y_true = y_true
        self.y_pred = y_pred
        self.model = model

    def accuracy(self) -> float:
        """
        Accuracy is the fraction of predictions our model got right.
        Args:
            y_true: np.ndarray
            y_pred: np.ndarray
        Returns:
            accuracy: float
        """
        try:
            logging.info("Entered the accuracy method of the Evaluation class")
            acc = accuracy_score(self.y_true, self.y_pred)
            logging.info(f"The accuracy value is: {acc}")
            return acc
        except Exception as e:
            logging.error(f"Exception in accuracy method: {e}")
            raise

    def precision(self, average: str = 'macro') -> float:
        """
        Precision is the ratio of correctly predicted positive observations to the total predicted positives.
        Args:
            y_true: np.ndarray
            y_pred: np.ndarray
            average: str - averaging method for multiclass classification
        Returns:
            precision: float
        """
        try:
            logging.info("Entered the precision method of the Evaluation class")
            prec = precision_score(self.y_true, self.y_pred, average=average)
            logging.info(f"The precision value is: {prec}")
            return prec
        except Exception as e:
            logging.error(f"Exception in precision method: {e}")
            raise

    def recall(self, average: str = 'macro') -> float:
        """
        Recall is the ratio of correctly predicted positive observations to all observations in actual class.
        Args:
            y_true: np.ndarray
            y_pred: np.ndarray
            average: str - averaging method for multiclass classification
        Returns:
            recall: float
        """
        try:
            logging.info("Entered the recall method of the Evaluation class")
            rec = recall_score(self.y_true, self.y_pred, average=average)
            logging.info(f"The recall value is: {rec}")
            return rec
        except Exception as e:
            logging.error(f"Exception in recall method: {e}")
            raise

    def f1(self, average: str = 'macro') -> float:
        """
        F1 Score is the weighted average of Precision and Recall.
        Args:
            y_true: np.ndarray
            y_pred: np.ndarray
            average: str - averaging method for multiclass classification
        Returns:
            f1_score: float
        """
        try:
            logging.info("Entered the f1 method of the Evaluation class")
            f1 = f1_score(self.y_true, self.y_pred, average=average)
            logging.info(f"The F1 score is: {f1}")
            return f1
        except Exception as e:
            logging.error(f"Exception in f1 method: {e}")
            raise

    def confusion_matrix(self) -> np.ndarray:
        """
        Confusion matrix is a summary of prediction results on a classification problem.
        Args:
            y_true: np.ndarray
            y_pred: np.ndarray
        Returns:
            confusion_matrix: np.ndarray
        """
        try:
            logging.info("Entered the confusion_matrix method of the Evaluation class")
            cm = confusion_matrix(self.y_true, self.y_pred)
            logging.info(f"Confusion Matrix:\n{cm}")
            return cm
        except Exception as e:
            logging.error(f"Exception in confusion_matrix method: {e}")
            raise