import logging

import pandas as pd
from .components.component_ingest_class import IngestData
from zenml.steps import step
import os 

FILE_NAME = os.path.basename(__file__)

@step()
def step_ingest_data(path) -> pd.DataFrame:
    """
    This function represent the step to ingest the data. It use instance from the IngestData class.
    Args:
        path: path to your data.
    Returns:
        df: pd.DataFrame containing the data
    """
    try:
        logging.info(f"Starting the {FILE_NAME}")
        ingest_data_instance = IngestData(path)
    except Exception as e:
        logging.error(f"An error has occure before starting {FILE_NAME}: {e}")
        raise e

