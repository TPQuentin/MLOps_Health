from steps.step_ingest_data import process_ingest_data
from zenml import pipeline


@pipeline
def mypipe():
    path = r"C:\Users\quentin.plourdeau\OneDrive - Tune Protect Group\Desktop\Youtube_Tutorial\MLOps_Health_Insurance\data\health.csv"
    process_ingest_data(path)


if __name__ == "__main__":
    mypipe()
