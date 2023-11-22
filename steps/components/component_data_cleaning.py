import logging
import pandas as pd
import os
from pathlib import Path

FILE_NAME = os.path.basename(__file__)

class DataCleaning:
    """
    Define class to clean the data
    """

    def __init__(self, data: pd.DataFrame) -> None:
        """Initialize the DataCleaning class"""
        self.data = data
    
    def clean_data(self) -> pd.DataFrame:
        """
        This method aims to clean the data of the instance class DataCleaning. Lie removing unused columns, filling-up missing value, etc
        Args:
            data: pd.DataFrame containing the data to clean
        Returns;
            clean_def: pd.Dataframe containing the cleaned data.
        """
        col_to_keep = ['Gender', 'Blood Type', 'Medical Condition', 'Admission Type', 'Test Results']
        try:
            logging.info(f"Start clean_data function from {FILE_NAME}")
            df = self.data[col_to_keep]
            clean_df = df.fillna('Mode')
            logging.info("Exit the function clean_data from {FILE_NAME} and return the clean dataframe")
            return clean_df

        except Exception as e:
            logging(f"Error while cleaning data with {FILE_NAME}: {e}")
            raise e
       