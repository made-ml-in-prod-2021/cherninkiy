import pandas as pd
from omegaconf import OmegaConf

from src.data.utils import read_dataset, split_dataset


def test_read_dataset(params: OmegaConf):

    df = read_dataset(params.data.data_path)

    assert isinstance(df, pd.DataFrame)
    assert (303, 14) == df.shape


def test_split_dataset(params: OmegaConf):

    df = read_dataset(params.data.data_path)

    train_size = params.data.train_size
    train, test = split_dataset(df, train_size)

    assert isinstance(train, pd.DataFrame)
    assert isinstance(test, pd.DataFrame)

    assert len(train) >= len(df) * train_size or len(test) <= len(df) * train_size
    assert len(train) + len(test) == len(df)

