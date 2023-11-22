import logging

import pandas as pd

import numpy as np

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.base import ClassifierMixin

class Evaluation:
    def __init__(self, model: ClassifierMixin , X_test: pd.DataFrame, y_test: pd.Series ) -> None:
        pass