import os
import sys
import click
import logging
import pandas as pd

from ml_pipeline.src.data.utils import read_dataset, split_dataset

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


@click.command("airflow-preprocess")
@click.option("--input-dir")
@click.option("--output-dir")
def preprocess(input_dir: str, output_dir: str):

    logger.info(f'Preprocessing data from {input_dir}...')

    df = read_dataset(f"{input_dir}/data.csv")
    target_df = pd.read_csv(f"{input_dir}/target.csv", names=["target"])
    df["target"] = target_df["target"]

    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(f"{output_dir}/data.csv", index=False, mode='a')
    logger.info(f"Train dataset saved to {output_dir}/data.csv")

    logger.info("Preprocessing data successed")


if __name__ == '__main__':
    preprocess()
