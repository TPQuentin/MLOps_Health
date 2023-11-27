import os
import pickle
from typing import Any, Type, Union

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from zenml.io import fileio

from zenml.materializers.base_materializer import BaseMaterializer

DEFAULT_FILENAME = "HealtcareInusranceionEnv"


class cs_materializer(BaseMaterializer):

    """
    Custom materializer for the Healtcare Inurance Project
    """

    ASSOCIATED_TYPES = (
        str,
        np.ndarray,
        pd.Series,
        pd.DataFrame,
        CatBoostClassifier,
        RandomForestClassifier,
        LGBMClassifier,
        XGBClassifier,
    )

    # Overwrite the load function from BaseMaterializer
    def load(self, data_type: type[Any]) -> Union[str, np.ndarray, pd.Series, pd.DataFrame, CatBoostClassifier, RandomForestClassifier, LGBMClassifier, XGBClassifier]:
        """Write logic here to load the data of an artifact.

        Args:
            data_type: The type of the model to be loaded.

        Returns:
            The data of the artifact.
        """
        super().load(data_type)

        filepath = os.path.join(self.artifact.uri, DEFAULT_FILENAME)
        with fileio.open(filepath, "rb") as fid:
            obj = pickle.load(fid)
        return obj

    def save(self, obj: Union[str, np.ndarray, pd.Series, pd.DataFrame, CatBoostClassifier, RandomForestClassifier, LGBMClassifier, XGBClassifier]) -> None:
        """Write logic here to save the data of an artifact.

        Args:
            data: The data of the artifact to save.
        """

        super().save(obj)
        filepath = os.path.join(self.artifact.uri, DEFAULT_FILENAME)
        with fileio.open(filepath, "wb") as fid:
            pickle.dump(obj, fid)
