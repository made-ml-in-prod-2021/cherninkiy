import os
import sys
import click
import logging
import joblib

from ml_pipeline.src.data.utils import read_dataset

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


@click.command("airflow-validate")
@click.option("--data-path")
@click.option("--model-path")
@click.option("--output-path")
def make_preds(data_path: str, model_path: str, output_path: str):

    logger.info(f"Making predictions...")
    logger.debug(f"data_path={data_path}")
    logger.debug(f"model_path={model_path}")
    logger.debug(f"output_path={output_path}")

    df = read_dataset(data_path)
    X = df.drop(columns=["target"])

    model = joblib.load(model_path)
    logger.info(f"Predictor model: {model}")

    df["target"] = model.predict(X)

    os.makedirs(output_path, exist_ok=True)
    df[["target"]].to_csv(f"{output_path}/target.csv", index=False)
    logger.info(f"Predictions saved to {output_path}/target.csv")

    logger.info("Making predictions successed")


if __name__ == '__main__':
    make_preds()
