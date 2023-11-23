import logging

import pandas as pd
from components.component_data_cleaning import DataCleaning


def get_data_for_test():
    try:
        df = pd.read_csv(
            r"C:\Users\quentin.plourdeau\OneDrive - Tune Protect Group\Desktop\Youtube_Tutorial\MLOps_Health_Insurance\data\health.csv")
        df = df.sample(n=100)
        data_clean = DataCleaning(df)
        df = data_clean.clean_data()
        df.drop(["Test Results"], axis=1, inplace=True)
        result = df.to_json(orient="split")
        return result
    except Exception as e:
        logging.error(e)
        raise e
