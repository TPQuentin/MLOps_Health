import logging
import pandas as pd
from zenml import step
import os
from components.component_ingest_class import IngestData

FILE_NAME = os.path.basename(__file__)


@step
def process_ingest_data(path: str) -> pd.DataFrame:
    """
    This function represents the step to ingest the data using an instance from the IngestData class.
    Returns:
        DataFrame: DataFrame containing the data.
    """

    # Call the get_data method to perform data ingestion
    try:
        logging.info(f"Starting {FILE_NAME}")
        # Create the IngestData instance
        ingest_data_instance = IngestData(path)
        data = ingest_data_instance.get_data()
        logging.info(f"Ending {FILE_NAME}")
        return data

    except Exception as e:
        logging.error(f"An error occurred while ingesting data: {e}")
        raise e
