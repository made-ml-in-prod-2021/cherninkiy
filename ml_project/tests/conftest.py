import os
import pytest
import numpy as np
from typing import NoReturn, Tuple
from omegaconf import OmegaConf
from hydra.experimental import compose, initialize

from src.data.utils import read_dataset, split_dataset
from src.features.build_features import FeatureBuilder, TargetBuilder


def load_conf():
    with initialize(config_path="../tests"):
        conf = compose(config_name="conftest")
    return conf

def load_train_test(conf: OmegaConf):
    df = read_dataset(conf.data.data_path)
    return split_dataset(df, conf.data.train_size)


@pytest.fixture(scope="session")
def params() -> OmegaConf:
    return load_conf()


@pytest.fixture(scope="session")
def train_data() -> Tuple[np.array, np.array]:
    conf = load_conf()

    train, _ = load_train_test(conf)
    X_train = FeatureBuilder(conf.features).fit_transform(train)
    y_train = TargetBuilder(conf.features).fit_transform(train)
    return X_train, y_train


@pytest.fixture(scope="session")
def test_data() -> Tuple[np.array, np.array]:
    conf = load_conf()

    train, test = load_train_test(conf)
    X_train = FeatureBuilder(conf.features).fit(train).transform(test)
    y_train = TargetBuilder(conf.features).fit(train).transform(test)
    return X_train, y_train


@pytest.fixture(scope="session")
def pipeline_params():
    with initialize(config_path="../tests"):
        conf = compose(config_name="pipeline")
    return conf