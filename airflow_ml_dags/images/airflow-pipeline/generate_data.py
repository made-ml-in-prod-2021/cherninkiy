import os
import click
import joblib
import logging
import pandas as pd
from omegaconf import OmegaConf
from hydra.experimental import compose, initialize

from ml_pipeline.src.data.generate import generate_features

logger = logging.getLogger("ml-pipeline/generate_data")


def load_conf(conf_path: str) -> OmegaConf:
    conf_dir, conf_name = os.path.split(conf_path)
    conf_name = ".".join(conf_name.split('.')[:-1])
    with initialize(config_path=conf_dir):
        params = compose(config_name=conf_name)
    return params


@click.command("generate")
@click.argument("output_dir")
@click.argument("config_path", default="conf/generator.yaml")
def generate_data(output_dir : str, config_path : str):

    logger.info("Synthesize data generation...")

    logger.info(f"config_path={config_path}")
    conf = load_conf(config_path)

    batch_size = conf.generator.batch_size
    logger.info(f"batch_size={batch_size}")
    random_state = conf.generator.random_state
    logger.info(f"random_state={random_state}")

    model_name = conf.generator.model_name
    logger.info(f"model_name={model_name}")
    model_path = conf[model_name].path
    logger.info(f"model_path={model_path}")

    model = joblib.load(model_path)
    logger.info(f"model_path={model}")

    X = generate_features(batch_size, random_state)
    y = pd.Series(model.predict(X))
    logger.info("Synthesize data generate successfully")

    os.makedirs(output_dir, exist_ok=True)

    X.to_csv(f"{output_dir}/data.csv", index=False, mode='a')
    logger.info(f"Features save to {output_dir}/data.csv")

    y.to_csv(f"{output_dir}/target.csv", index=False)
    logger.info(f"Target save to {output_dir}/target.csv", mode='a')


if __name__ == "__main__":
    generate_data()