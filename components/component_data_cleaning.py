import logging
import pandas as pd
import os
from typing import Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

FILE_NAME = os.path.basename(__file__)


class DataCleaning:
    """
    Class to clean the data.
    """

    def __init__(self, data: pd.DataFrame) -> None:
        """Initialize the DataCleaning class with a DataFrame."""
        self.data = data

    def clean_data(self) -> pd.DataFrame:
        """
        Clean the data by removing unused columns and filling up missing values.

        Returns:
            pd.DataFrame: The cleaned data.
        """
        col_to_keep = ['Age', 'Billing Amount', 'Test Results']
        try:
            logging.info(f"Starting clean_data() {FILE_NAME}")
            df = self.data[col_to_keep]

            # Example: Fill missing values with the mode of each column
            for col in df.columns:
                df.loc[:, col] = df[col].fillna(df[col].mode()[0])

            logging.info(f"Ending clean_data() {FILE_NAME}")
            return df

        except Exception as e:
            logging.error(f"Error while cleaning data with {FILE_NAME}: {e}")
            raise e

    def divide_data(self, clean_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Divide the data into train and test sets.

        Args:
            clean_df (pd.DataFrame): The cleaned DataFrame containing features and labels.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: Training and testing features and labels.
        """
        try:
            logging.info(f"Starting divide_data() {FILE_NAME}")
            X = clean_df.drop(['Test Results'], axis=1)
            y = clean_df['Test Results']

            # Encoding the 'Test Results' column if it is categorical
            if y.dtype == 'object':
                encoder = LabelEncoder()
                y = encoder.fit_transform(y)
                y = pd.Series(y)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42)

            logging.info(f"Ending divide_data() {FILE_NAME}")
            return X_train, X_test, y_train, y_test

        except Exception as e:
            logging.error(
                f"Error while running method divide_data from {FILE_NAME}: {e}")
            raise e
