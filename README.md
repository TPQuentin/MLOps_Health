# MLOps

Use health data to predict the result of a test (Normal, Abnormal, Incnclusive).

The main objective is to understand how ZenML can be leverage to develop and build production-ready ML pipeline to predict the previous result.

Zenml enable to integrate tools by defining custom stack component to run pipeline. Here MLflow will be use for tracking, registering and deploying ML model locally.

## Python Requirements

```bash
git clone https://github.com/TPQuentin/MLOps_Health.git
cd MLOps_Health_Insurance
python -m venv healthenv

source \healthenv\Scripts\activate

pip install -r requirements.txt
```

Now building zenml server

```bash
zenml init
pip install "zenml[server]"
```

Check if the server is well build:

```bash
zenml server list
zenml up
```

Install the zenml integrations:

```bash
zenml integration install mlflow -y
```

Set-up a custom stack to run the training pipeline and deployment pipeline (model-deployer: mlflow_model_deployer,
experiment-tracker: mlflow_experiment_tracker,
artifact-store: default,
orchestrator: default,
model-registry: mflow_model_register):

```bash
 zenml stack register my_stack_name -d mlflow_model_deployer -e mlflow_experiment_tracker -a default -o default -r mlflow_model_register
 zenml stack set my_stack_name
 zenml stack list
 zenml stack describe my_stack_name
```

## Run the pipelines

Training pipeline:

```
python run_training_pipeline.py
```

Deployment pipeline:

```
python run_deployment_pipeline.py
```

## Run the streeamlit app

```bash
streamlit run streamlit_app.py
```
