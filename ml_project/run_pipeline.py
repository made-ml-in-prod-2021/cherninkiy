from typing import NoReturn
from os import path
import logging
import click
import hydra
from omegaconf import OmegaConf
from hydra.experimental import compose, initialize

from src.train_pipeline import train_pipeline
from src.predict_pipeline import predict_pipeline


logger = logging.getLogger("main")


def load_conf(conf_path: str) -> OmegaConf:
    conf_dir, conf_name = path.split(conf_path)
    conf_name = ".".join(conf_name.split('.')[:-1])
    with initialize(config_path=conf_dir):
        params = compose(config_name=conf_name)
    return params

@click.group(invoke_without_command=True, no_args_is_help=True)
@click.pass_context
def main(ctx) -> NoReturn:
    pass


@main.command()
@click.argument('config')
def train(config: str) -> NoReturn:
    params = load_conf(config)
    train_pipeline(params)


@main.command()
@click.argument('config')
def predict(config: str) -> NoReturn:
    params = load_conf(config)
    print(predict_pipeline(params))


if __name__ == "__main__":
    main()