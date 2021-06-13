import click
import logging
from os import path
from typing import NoReturn
from hydra import compose, initialize_config_dir

from ml_pipeline.src.entities.pipeline_params import PipelineParams
from ml_pipeline.src.train_pipeline import train_pipeline
from ml_pipeline.src.predict_pipeline import predict_pipeline

logger = logging.getLogger("ml-pipeline")


def load_conf(conf_path: str) -> PipelineParams:
    conf_dir, conf_name = path.split(conf_path)
    conf_name = ".".join(conf_name.split('.')[:-1])
    with initialize_config_dir(config_dir=conf_dir):
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
    preds = predict_pipeline(params)
    print("\n".join(map(str, preds)))


if __name__ == "__main__":
    main()
