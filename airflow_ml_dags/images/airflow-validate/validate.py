import os
import sys
import click
import logging
import joblib
import json
from sklearn.metrics import f1_score, accuracy_score

from ml_pipeline.src.data.utils import read_dataset

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


@click.command("airflow-validate")
@click.option("--data-path")
@click.option("--model-path")
def validate(data_path: str, model_path: str):

    logger.info(f"Validating pipeline...")
    logger.debug(f"data_path={data_path}")
    logger.debug(f"model_path={model_path}")

    df = read_dataset(f"{data_path}/test.csv")
    X = df.drop(columns=["target"])
    y = df["target"]

    model = joblib.load(model_path)
    logger.info(f"Predictor model: {model}")

    preds = model.predict(X)

    metrics = {
        'f1_score': f1_score(y, preds),
        'accuracy_score': accuracy_score(y, preds)
    }

    with open(f"{data_path}/metrics.txt", "w") as fout:
        fout.write(str(metrics))
    logger.info(f"Metrics saved into {data_path}/metrics.txt")

    logger.info("Validating pipeline successed")


if __name__ == '__main__':
    validate()
