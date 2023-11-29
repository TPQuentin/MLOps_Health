import numpy as np
import pandas as pd
import streamlit as st
from run_deployment_pipeline import run_deployment_pipeline
from pipelines.deployment_pipeline import process_prediction_service_loader
from zenml.integrations.mlflow.services import MLFlowDeploymentService


def main():
    st.title("Pipeline with ZenML")

    age = st.number_input("Age", step=1)
    # age = st.sidebar.slider(
    # "age", min_value=0, max_value=125)
    Billing_Amount = st.number_input("Billing Amount")
    # Billing_Amount = st.sidebar.slider(
    # "Billing Amount", min_value=0, max_value=1000000)

    if st.button("Predict"):
        service = process_prediction_service_loader(
            pipeline_name="continuous_deployment_pipeline",
            pipeline_step_name="model_deployer",
            running=True,
        )
        if service is None:
            st.write(
                "No service could be found. The pipeline must be run first to create a service."
            )
            run_deployment_pipeline()

        df = pd.DataFrame(
            {
                "Age": [age],
                "Billing Amount": [Billing_Amount]
            }
        )
        data = df.values()

        pred = service.predict(data)
        st.success(
            f"The prediction for the Test result is :-{pred}"
        )


if __name__ == "__main__":
    main()
