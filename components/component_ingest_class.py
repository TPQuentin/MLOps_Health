import logging
import pandas as pd
from pathlib import Path
from typing import Union
import os

FILE_NAME = os.path.basename(__file__)


class IngestData:
    """
    Data Ingestion class which ingests data from the source and returns a pd.DataFrame
    """

    def __init__(self, path: str) -> None:
        """Initialize the data ingestion class with a required path."""
        self.path = path

    def get_data(self) -> pd.DataFrame:
        """
        Fetch data from the indicated path.

        Returns:
            pd.DataFrame: The DataFrame containing the ingested data.
        """

        try:
            logging.info(
                f"Start IngestData.get_data() from {FILE_NAME}")
            df = pd.read_csv(self.path)
            return df
        except Exception as e:
            logging.error(f"An error has occured when calling get_data(): {e}")
            raise e
