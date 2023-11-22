import logging

import pandas as pd
from pathlib import Path
from typing import Union
import os 

FILE_NAME = os.path.basename(__file__)

class IngestData:
    """
    Data Ingestion class which ingest data from the source and returns a DataFrame
    """

    def __init__(self, path: str = None) -> None:
        """Initialize the data ingestion class"""
        self.path = path

    def get_data(self, path: Union[str, Path] ) -> pd.DataFrame:
        """
        This funnction enable to fetch the data from the indicated path.
        Args:
            path: The path of the source
        Return:
            df: The DataFrame containing the ingested data
        """
        logging.info(f"Start IngestData.get_data() from script {FILE_NAME}")
        if isinstance(path, str):
            path = Path(path)
        df = pd.read_csv(path)
        return df