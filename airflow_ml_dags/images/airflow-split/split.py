import sys
import click
import logging

from ml_pipeline.src.data.utils import read_dataset, split_dataset

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


@click.command("airflow-split")
@click.option("--data-path")
@click.option("--train-size", default=0.7)
@click.option("--random-state", default=None)
def split_train_test_data(data_path: str, train_size: float, random_state: int):

    logger.info(f'Splitting data from {data_path}...')

    df = read_dataset(f"{data_path}/data.csv")

    train, test = split_dataset(df, train_size, random_state)

    train.to_csv(f"{data_path}/train.csv", index=False, mode='a')
    logger.info(f"Train dataset saved to {data_path}/train.csv")

    test.to_csv(f"{data_path}/test.csv", index=False, mode='a')
    logger.info(f"Train dataset saved to {data_path}/test.csv")

    logger.info("Splitting data successed")

if __name__ == '__main__':
    split_train_test_data()
