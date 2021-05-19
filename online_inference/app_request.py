import json
import requests
import pandas as pd

DATA_PATH = "data/inference_test/data.csv"


if __name__ == "__main__":
    data = pd.read_csv(DATA_PATH) \
        .drop("target", axis=1) \
        .to_dict(orient="records")
    print("Features sample:")
    print(data[0])
    response = requests.post(
        "http://0.0.0.0:8000/predict",
        json.dumps(data)
    )
    print("Target:")
    print(f"Status code: {response.status_code}")
    print(response.json()[0])