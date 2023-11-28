import click
from pipelines.deployment_pipeline import (continuous_deployment_pipeline,
                                           process_dynamic_importer,
                                           inference_pipeline,
                                           process_prediction_service_loader,
                                           process_prediction
                                           )

from rich import print
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import MLFlowModelDeployer


@click.command()
@click.option(
    "--min-accuracy",
    default=0.1,
    help="minimum accuracy to trigger the deployment.",
)
@click.option(
    "--stop-service",
    is_flag=True,
    default=False,
    help="Stop the prediction service.",
)
# We have to put the click option arguments in the input to parse them automaically
def run_deployment_pipeline(min_accuracy: float, stop_service: bool):
    """Run the mlflow example pipeline"""
    if stop_service:
        # Get the MLFow model deployer stack component
        model_deployer = MLFlowModelDeployer.get_active_model_deployer()

        # fetch existing services with the same pipeline name, step name and model name
        service = model_deployer.find_model_server(
            pipeline_name="continuous_deployment_pipeline",
            pipeline_step_name="model_deployer",
            running=True
        )

        if service:
            service.stop(timeout=10)
        return

    deployment = continuous_deployment_pipeline.with_options(
        config_path="deploy_config.yaml")
    deployment(min_accuracy)

    inference = inference_pipeline(
        process_dynamic_importer(),
        process_prediction_service_loader(
            pipeline_name="continuous_deployment_pipeline",
            step_name="model_deployer"
        ),
        predictor=process_prediction()
    )
    inference()

    print(
        "Now run \n "
        f"    mlflow ui --backend-store-uri {get_tracking_uri()}\n"
        "To inspect your experiment runs within the mlflow UI.\n"
        "You can find your runs tracked within the `mlflow_example_pipeline`"
        "experiment. Here you'll also be able to compare the two runs.)"
    )

    model_deployer = MLFlowModelDeployer.get_active_model_deployer()

    service = model_deployer.find_model_server(
        pipeline_name="continuous_deployment_pipeline",
        pipeline_step_name="model_deployer",
        running=True
    )

    if service[0]:
        print(
            f"The MLflow prediction server is running locally as a daemon process "
            f"and accepts inference requests at:\n"
            f"    {service[0].prediction_url}\n"
            f"To stop the service, re-run the same command and supply the "
            f"`--stop-service` argument."
        )


if __name__ == "__main__":
    run_deployment_pipeline()
