import pandas as pd
from typing import Tuple
from sklearn.model_selection import train_test_split


def read_dataset(
            data_path: str
    ) -> pd.DataFrame:
    """
    Read csv data from file.
    Parameters
    ----------
    path: str
        Path to dataset csv file.
    """
    data = pd.read_csv(data_path)
    return data


def split_dataset(
            data: pd.DataFrame,
            train_size: int = 0.7,
            random_state: int = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split dataset to train and test parts.
    Parameters
    ----------
    data: pd.DataFrame
        Dataset to split.
    params: OmegaConf
        Model params.
    """
    train, test = train_test_split(
            data,
            train_size=train_size,
            random_state=random_state,
            shuffle=True
    )
    return train, test