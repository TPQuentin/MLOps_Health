import logging
import pandas as pd

from zenml.steps import step
import os
from pathlib import Path
from .components.component_data_cleaning import DataCleaning
FILE_NAME = os.path.basename(__file__)

@step()
def process_clean_data(data: pd.DataFrame)->pd.DataFrame:

    try:
        logging.info(f"Start of {FILE_NAME}")
        cleaner = DataCleaning(data)
        cleaned_data = cleaner.clean_data()
        logging.info(f"Enf of {FILE_NAME}")

        return cleaned_data
    except Exception as e:
        logging.error(f"Error for {FILE_NAME}: {e}")
        raise e
    return cleaned_data
    
