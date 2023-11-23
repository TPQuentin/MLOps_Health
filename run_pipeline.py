from pipelines.training_pipeline import train_pipeline
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri

from steps.step_ingest_data import process_ingest_data
from steps.step_clean_data import process_clean_data
from steps.step_train_model import process_train_model
from steps.step_evaluation import process_evaluate


def run_pipeline():
    train_pipeline(process_ingest_data(), process_clean_data(),
                   process_train_model(), process_evaluate())

    train_pipeline.run()


if __name__ == "__main__":

    print("Run the pipelinne")
    run_pipeline()

    print(
        "Now run \n "
        f"    mlflow ui --backend-store-uri '{get_tracking_uri()}'\n"
        "To inspect your experiment runs within the mlflow UI.\n"
        "You can find your runs tracked within the `mlflow_example_pipeline`"
        "experiment. Here you'll also be able to compare the two runs.)"
    )
