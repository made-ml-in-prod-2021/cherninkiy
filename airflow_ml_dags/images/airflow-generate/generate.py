import os
import sys
import logging
import click
import joblib
import pandas as pd

from ml_pipeline.src.data.generate import generate_features

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


@click.command("generate")
@click.option("--output-path")
@click.option("--model-path")
@click.option("--batch-size", default=1000)
@click.option("--random-state", default=None)
def generate_data(
        output_path: str,
        model_path: str,
        batch_size: int,
        random_state:int
) -> pd.DataFrame:

    logger.info("Data generation...")

    logger.debug(f"model_path={model_path}")
    logger.debug(f"batch_size={batch_size}")
    logger.debug(f"random_state={random_state}")

    model = joblib.load(model_path)
    logger.info(f"Synthesize model: {model}")

    X = generate_features(batch_size, random_state)
    y = pd.Series(model.predict(X))

    os.makedirs(output_path, exist_ok=True)

    X.to_csv(f"{output_path}/data.csv", index=False, mode='a')
    logger.info(f"Features saved to {output_path}/data.csv")

    y.to_csv(f"{output_path}/target.csv", index=False, mode='a')
    logger.info(f"Target saved to {output_path}/target.csv")


if __name__ == '__main__':
    generate_data()
