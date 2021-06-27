import os
import sys
import click
import joblib
import logging
import pandas as pd

from ml_pipeline.src.data.utils import read_dataset
from ml_pipeline.src.entities.model_params import ModelParams
from ml_pipeline.src.models.train_model import train_model

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


@click.command("airflow-train")
@click.option("--data-path")
@click.option("--model-name")
@click.option("--model-path")
def train_pipeline(data_path: str, model_name: str, model_path: str):

    logger.info(f"Pipeline training ...")
    logger.debug(f"data_path={data_path}")
    logger.debug(f"model_name={model_name}")
    logger.debug(f"model_path={model_path}")

    df = read_dataset(f"{data_path}/train.csv")
    X = df.drop(columns=["target"])
    y = df["target"]

    params = ModelParams(model=model_name, path=model_path)
    model = train_model(X, y, params)

    os.makedirs(os.path.split(model_path)[0], exist_ok=True)
    joblib.dump(model, model_path)
    logger.info(f"Model saved into {model_path}")


if __name__ == '__main__':
    train_pipeline()
