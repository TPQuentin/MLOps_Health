import json
import os

import numpy as np
import pandas as pd

from materializer.custom_materializer import cs_materializer

from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import MLFlowModelDeployer
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml import pipeline
from zenml.steps import step, Output, BaseParameters

from .utils import get_data_for_test

from steps.step_ingest_data import process_ingest_data
from steps.step_clean_data import process_clean_data
from steps.step_train_model import process_train_model
from steps.step_evaluation import process_evaluate


requirement_file = os.path.join(os.path.dirname(__file__), "requirements.txt")


@step(enable_cache=True, output_materializers=cs_materializer)
def process_dynamic_importer() -> Output(data=str):
    # The output of this step will use a custom materializer cs_materializer to output a model pickle file. The materializer enable to serializer and deserializer custom python type using a pickle file. Json could have been use also for example.
    data = get_data_for_test()
    # The steps will automatically use the cs_materialize class to save and load the model in the context of ZenML
    return data


# In the context of Zenml the BaseParameter class is used to pass a parameter from one step to another.
class DeploymentTriggerConfig(BaseParameters):
    """Parameters that are used to trigger the deployment"""

    min_accuracy: float


@step
def process_deployment_trigger(accuracy: float, config: DeploymentTriggerConfig) -> bool:
    """
    Args:
        accuracy: accuracy of the trained model
        config: Config of the DeploymentTrigger
    Returns:
        True if the accuracy > config.min_accuracy else false
    """
    return accuracy > config.min_accuracy


class MLFlowDeploymentLoaderStepConfig(BaseParameters):
    """MLflow deployment getter parameters

    Attributes:
        pipeline_name: name of the pipeline that deployed the MLflow prediction
            server

        step_name: the name of the step that deployed the MLflow prediction
            server

        running: when this flag is set, the step only returns a running service

        model_name: the name of the model that is deployed
    """

    pipeline_name: str
    step_name: str
    running: bool
    model_name: str


@step
def process_prediction_service_loader(config: MLFlowDeploymentLoaderStepConfig) -> MLFlowModelDeployer:
    """
    Get the prediction service started by the deployment pipeline
    """

    # Get the MLFow model deployer stack component
    model_deployer = MLFlowModelDeployer.get_active_model_deployer()

    # fetch existing services with the same pipeline name, step name and model name
    existing_services = model_deployer.find_model_server(
        pipeline_name=config.pipeline_name,
        step_name=config.step_name,
        running=config.running
    )
    if not existing_services:
        raise RuntimeError(
            f"No MLflow prediction service deployed by the "
            f"{config.step_name} step in the {config.pipeline_name} "
            f"pipeline is currently "
            f"running."
        )

    return existing_services[0]


@step
def process_prediction(service: MLFlowDeploymentService, data: str) -> np.ndarray:
    """
    Run an inference pipeline against a prediction service

    Args:
        service: the service loaded
        data: str because json file
    Return:
        prediction: the prediction for the data
    """
    service.start(timeout=10)
    data = json.loads(data)

    columns = data["columns"]
    list_data = data["data"]
    df = pd.DataFrame(data=list_data, columns=columns)

    arr = df.values()
    prediction = service.predict(arr)

    return prediction


@pipeline(enable_cache=False)
def continuous_deployment_pipeline(process_ingest_data, process_clean_data, process_train_model, process_evaluate, process_deployment_trigger, model_deployer):
    df = process_ingest_data()

    X_train, X_test, y_train, y_test = process_clean_data(df)

    model = process_train_model(X_train, X_test, y_train, y_test)

    accuracy, _, _, _ = process_evaluate(model, X_test, y_test)

    deploy_decision = process_deployment_trigger(accuracy=accuracy)

    model_deployer(deploy_decision, model)


@pipeline
def inference_pipeline(process_dynamic_importer, process_prediction_service_loader, process_prediction):
    batch_data = process_dynamic_importer()
    model_deployment_service = process_prediction_service_loader()
    process_prediction(model_deployment_service, batch_data)
