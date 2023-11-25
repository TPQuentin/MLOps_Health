from steps.step_clean_data import process_clean_data
from zenml import pipeline
import pandas as pd


@pipeline
def mycleanpipe():
    print("df is here")
    # path = r"C:\Users\quentin.plourdeau\OneDrive - Tune Protect Group\Desktop\Youtube_Tutorial\MLOps_Health_Insurance\data\health.csv"
    # data = pd.read_csv(path)

    process_clean_data(data)


if __name__ == "__main__":
    mycleanpipe()
