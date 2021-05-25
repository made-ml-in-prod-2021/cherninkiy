import os
import json
import requests
import pandas as pd

DATA_PATH = "data/test.csv"
PREDICTOR_HOST = os.environ.get("HOST", default="0.0.0.0")
PREDICTOR_PORT = os.environ.get("PORT", default=8080)


def make_request():
    data = pd.read_csv(DATA_PATH) \
        .drop("target", axis=1) \
        .to_dict(orient="records")

    response = requests.post(
        f"http://{PREDICTOR_HOST}:{PREDICTOR_PORT}/predict",
        json.dumps(data)
    )

    print("Predictor response:")
    print(f"Status code: {response.status_code}")
    resp = response.json()
    if resp is not None:
        print(list(f | t for (f, t) in zip(data, resp)))


if __name__ == "__main__":
    make_request()