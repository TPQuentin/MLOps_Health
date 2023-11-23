import logging
import pandas as pd

from zenml.steps import step
import os
from components.component_data_cleaning import DataCleaning
from typing import Tuple, Annotated

FILE_NAME = os.path.basename(__file__)


@step
def process_clean_data(data: pd.DataFrame) -> Tuple[Annotated[pd.DataFrame, "Training Features"],
                                                    Annotated[pd.DataFrame,
                                                              "Testing Features"],
                                                    Annotated[pd.Series,
                                                    "Training Labels"],
                                                    Annotated[pd.Series, "Testing Labels"]]:

    try:
        logging.info(f"Start cleaning {FILE_NAME}")
        cleaner = DataCleaning(data)
        cleaned_data = cleaner.clean_data()
        logging.info(f"Enf of cleaning {FILE_NAME}")

        logging.info(f"Start splitting {FILE_NAME}")
        X_train, X_test, y_train, y_test = cleaner.divide_data(cleaned_data)
        logging.info(f"Enf of splitting {FILE_NAME}")

        return X_train, X_test, y_train, y_test

    except Exception as e:
        logging.error(f"Error for {FILE_NAME}: {e}")
        raise e
