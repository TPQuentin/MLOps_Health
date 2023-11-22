import logging
import pandas as pd
from pathlib import Path
from typing import Union
import os

FILE_NAME = os.path.basename(__file__)

class IngestData:
    """
    Data Ingestion class which ingests data from the source and returns a DataFrame
    """

    def __init__(self, path: Union[str, Path]) -> None:
        """Initialize the data ingestion class with a required path."""
        self.path = path

    #@staticmethod
    def get_data(self) -> pd.DataFrame:
        """
        Fetch data from the indicated path.
        
        Args:
            path (Union[str, Path]): The path of the source file.
        
        Returns:
            pd.DataFrame: The DataFrame containing the ingested data.
        """
        logging.info(f"Start IngestData.get_data() from script {FILE_NAME}")
        if isinstance(self.path, str):
            path = Path(self.path)

        if not self.path.exists() or not self.path.is_file():
            raise FileNotFoundError(f"File not found at {self.path}")

        # Check file extension and handle different formats if needed
        if path.suffix.lower() != '.csv':
            raise ValueError("Unsupported file format. Only CSV files are supported.")

        try:
            df = pd.read_csv(self.path)
            return df
        except pd.errors.EmptyDataError:
            raise ValueError(f"File at {self.path} is empty.")
