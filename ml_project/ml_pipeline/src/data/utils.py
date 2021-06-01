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
    data_path: str
        Path to dataset csv file.
    """
    data = pd.read_csv(data_path)
    return data


def split_dataset(
            data: pd.DataFrame,
            train_size: float = 0.7,
            random_state: int = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split dataset to train and test parts.
    Parameters
    ----------
    data: pd.DataFrame
        Dataset to split.
    train_size: float
        Size of training part.
    random_state: int
        Random state seed.
    """
    train, test = train_test_split(
            data,
            train_size=train_size,
            random_state=random_state,
            shuffle=True
    )
    return train, test