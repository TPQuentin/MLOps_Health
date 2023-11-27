from pipelines.training_pipeline import train_pipeline_test
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri

from steps.step_ingest_data import process_ingest_data
from steps.step_clean_data import process_clean_data
from steps.step_train_model import process_train_model
from steps.step_evaluation import process_evaluate


if __name__ == "__main__":

    print("Run the pipelinne")
    path = r"C:\Users\quentin.plourdeau\OneDrive - Tune Protect Group\Desktop\Youtube_Tutorial\MLOps_Health_Insurance\data\health.csv"

    train_pipeline_test(path)

    print(
        "Now run \n "
        f"    mlflow ui --backend-store-uri '{get_tracking_uri()}'\n"
        "To inspect your experiment runs within the mlflow UI.\n"
        "You can find your runs tracked within the `mlflow_example_pipeline`"
        "experiment. Here you'll also be able to compare the two runs.)"
    )
