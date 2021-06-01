import pytest
import numpy as np
import pandas as pd
from typing import Tuple
from omegaconf import OmegaConf
from hydra.experimental import compose, initialize

from ..src.data.utils import read_dataset, split_dataset
from ..src.features.build_features import FeatureBuilder, TargetBuilder


def load_conf() -> OmegaConf:
    with initialize(config_path=""):
        conf = compose(config_name="conftest")
    return conf


def load_train_test(conf: OmegaConf) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = read_dataset(conf.data.data_path)
    return split_dataset(df, conf.data.train_size)


def make_synthesize_data(
        batch_size : int,
        random_state: int = None,
) -> pd.DataFrame:
    """
    Generate synthetic data.
    Parameters
    ----------
    batch_size: int
        Size of data to generate.
    random_state: int
        Random state seed.
    """
    np.random.seed(random_state)
    return pd.concat((
        pd.Series(np.random.normal(55, 10, batch_size).astype(int), name='age'),
        pd.Series(np.random.binomial(1, 0.68, batch_size), name='sex'),
        pd.Series(np.random.choice(4, batch_size, p=[0.47, 0.16, 0.30, 0.07]), name='cp'),
        pd.Series(np.random.normal(130, 17, batch_size).astype(int), name="trestbps"),
        pd.Series(np.random.normal(240, 52, batch_size).astype(int), name="chol"),
        pd.Series(np.random.binomial(1, 0.15, batch_size), name="fbs"),
        pd.Series(np.random.choice(3, batch_size, p=[0.48, 0.50, 0.02]), name="restecg"),
        pd.Series(np.random.normal(153, 23, batch_size).astype(int), name="thalach"),
        pd.Series(np.random.binomial(1, 0.33, batch_size), name="exang"),
        pd.Series(np.random.binomial(1, 0.67, batch_size) \
              * (1.0 + np.random.normal(1.5, 1.0, batch_size)), name="oldpeak"),
        pd.Series(np.random.choice(3, batch_size, p=[0.08, 0.460, 0.46]), name="slope"),
        pd.Series(np.random.choice(5, batch_size, p=[0.58, 0.22, 0.12, 0.06, 0.02]), name="ca"),
        pd.Series(np.random.choice(4, batch_size, p=[0.01, 0.06, 0.38, 0.55]), name="thal"),
    ), axis=1)


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
def pipeline_params() -> OmegaConf:
    with initialize(config_path=""):
        conf = compose(config_name="pipeline")
    return conf


@pytest.fixture(scope="session")
def synthetic_data() -> pd.DataFrame:
    conf = load_conf()
    return make_synthesize_data(
        conf.synthetic.batch_size,
        conf.synthetic.random_state
    )