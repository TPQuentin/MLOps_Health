import logging
import pandas as pd
import os
from pathlib import Path
from typing import Union, Annotated, Tuple
from sklearn.model_selection import train_test_split

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
        col_to_keep = ['Age', 'Billing Amount', 'Test Results']
        try:
            logging.info(f"Start clean_data function from {FILE_NAME}")
            df = self.data[col_to_keep]
            clean_df = df.fillna('Mode')
            logging.info("Exit the function clean_data from {FILE_NAME} and return the clean dataframe")
            return clean_df

        except Exception as e:
            logging(f"Error while cleaning data with {FILE_NAME}: {e}")
            raise e
       
    @staticmethod
    def divide_data(df: pd.DataFrame) -> Tuple[Annotated[pd.DataFrame, "Training Features"],
                                                Annotated[pd.DataFrame, "Testing Features"],
                                                Annotated[pd.Series, "Training Labels"],
                                                Annotated[pd.Series, "Testing Labels"]]:
        """
        Divide the data into train and test data.

        Args:
            df (pd.DataFrame): The input DataFrame containing features and labels.

        Returns:
            Tuple of (X_train, X_test, y_train, y_test):
                - X_train (pd.DataFrame): Training features.
                - X_test (pd.DataFrame): Testing features.
                - y_train (pd.Series): Training labels.
                - y_test (pd.Series): Testing labels.
        """
        try:
            logging.info(f"Start divide_data from {FILE_NAME}")
            X = df.drop(['Test Results'], axis=1)  # Assuming 'Test Results' is the label column
            y = df['Test Results']  # Replace 'Test Results' with your actual label column name

            ###### NEEEEEEED TO ENCODE THE DATA HEEEEEERE

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            return X_train, X_test, y_train, y_test

        except Exception as e:
            logging.error(f"Error while running method divide_data from {FILE_NAME}: {e}")
            raise e