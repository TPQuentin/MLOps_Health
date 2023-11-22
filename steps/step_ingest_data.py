import logging
import pandas as pd
from .components.component_ingest_class import IngestData
from zenml.steps import step
import os
from pathlib import Path
from typing import Union

FILE_NAME = os.path.basename(__file__)

@step()
def step_ingest_data(path: Union[str, Path]) -> pd.DataFrame:
    """
    This function represents the step to ingest the data using an instance from the IngestData class.
    
    Args:
        path (str): Path to your data.
    
    Returns:
        pd.DataFrame: DataFrame containing the data.
    """
    logging.info(f"Starting {FILE_NAME}")
    
    # Create the IngestData instance
    ingest_data_instance = IngestData(path)
    
    # Call the get_data method to perform data ingestion
    try:
        df = ingest_data_instance.get_data(path)
        return df
    except Exception as e:
        logging.error(f"An error occurred while ingesting data: {e}")
        raise e
