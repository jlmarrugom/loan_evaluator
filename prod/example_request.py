import json
from typing import Tuple

import numpy as np
import pandas as pd
import requests

SERVICE_URL = "http://localhost:3000/classify"

def load_random_datapoint(data_path: str)->Tuple[np.ndarray, np.ndarray]:
    """Takes a random DataPoint from a dataset"""
    df = pd.read_csv(data_path)
    random_datapoint = df.sample(1).fillna(0)
    features = random_datapoint.drop('Loan_Approval', axis=1)
    expected_output = random_datapoint[['Loan_Approval']]
    return features.values, expected_output.values

def make_request_to_bento_service(service_url: str, input_array: np.ndarray)->str:
    """Make a post to the service with the model"""
    input_data = json.dumps(input_array.tolist())
    response = requests.post(
        service_url,
        data=input_data, 
        headers={"content-type": "application/json"}
        )
    return response.text

def main():
    input_array, expected_output = load_random_datapoint("train/dataset.csv")
    prediction = make_request_to_bento_service(SERVICE_URL, input_array)

    print(f"Prediction: {json.loads(prediction)}")
    print(f"Expected Output: {expected_output}")

if __name__ == "__main__":
    main()
