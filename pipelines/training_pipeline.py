import logging
from zenml.pipelines import pipeline
import os
from steps.config import ModelNameConfig

FILE_NAME = os.path.basename(__file__)


@pipeline
def train_pipeline(process_ingest_data, process_clean_data, process_train_model, process_evaluate):
    """
    Args:
        step_ingest_data: DataClass
        step_clean_data: DataClass
        step_model_train: DataClass
        step_evaluation: DataClass
    Returns:
        mse: float
        rmse: float
    """

    print("HIHIHIHHIHIHIHIHIHIHIHIHIHIHIHIh")
    logging.info(f"Starting {FILE_NAME}")
    df = process_ingest_data()

    x_train, x_test, y_train, y_test = process_clean_data(df)

    config = ModelNameConfig
    model = process_train_model(x_train, x_test, y_train, y_test, config)

    accuracy, precision, recall, f1 = process_evaluate(model, x_test, y_test)
    logging.info(f"Ending {FILE_NAME}")
