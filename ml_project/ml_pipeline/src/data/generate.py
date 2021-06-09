import numpy as np
import pandas as pd
from typing import Union
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

ClassifierModel = Union[LogisticRegression, RandomForestClassifier]


def generate_features(
        batch_size: int,
        random_state: int = None
) -> pd.DataFrame:
    """
    Synthetic features generator.
    Parameters
    ----------
    batch_size: int
        Batch size.
    random_state: int
        Random state seed.
    Returns
    -------
        Pandas dataframe with features values.
    """

    if random_state is not None:
        np.random.seed(random_state)

    # columns data generators
    age = pd.Series(np.random.normal(55, 10, batch_size).astype(int), name="age")
    sex = pd.Series(np.random.binomial(1, 0.68, batch_size), name="sex")
    cp = pd.Series(np.random.choice(4, batch_size, p=[0.47, 0.16, 0.30, 0.07]), name="cp")
    trestbps = pd.Series(
            np.hstack((
                110 + 10 * np.random.choice(4, batch_size // 4, p=[0.1, 0.3, 0.3, 0.3]),
                np.array((
                    np.random.normal(130, 40, batch_size - batch_size // 4).astype(int),
                    np.random.normal(110, 10, batch_size - batch_size // 4).astype(int),
                    np.random.normal(120, 10, batch_size - batch_size // 4).astype(int),
                    np.random.normal(130, 10, batch_size - batch_size // 4).astype(int),
                    np.random.normal(140, 10, batch_size - batch_size // 4).astype(int)
                )).mean(axis=0).astype(int)
            )), name="trestbps", index=np.random.permutation(list(range(batch_size)))
        ).sort_index()
    chol = pd.Series(np.random.normal(240, 50, batch_size).astype(int), name="chol")
    fbs = pd.Series(np.random.binomial(1, 0.15, batch_size), name="fbs")
    restecg = pd.Series(np.random.choice(3, batch_size, p=[0.48, 0.50, 0.02]), name="restecg")
    thalach = pd.Series(np.random.normal(150, 23, batch_size).astype(int), name="thalach")
    exang = pd.Series(np.random.binomial(1, 0.33, batch_size), name="exang")
    oldpeak = pd.Series(0.5 + np.random.normal(0.5, 1.0, batch_size), name="oldpeak").clip(lower=0.0)
    slope = pd.Series(np.random.choice(3, batch_size, p=[0.07, 0.46, 0.47]), name="slope")
    ca = pd.Series(np.random.choice(5, batch_size, p=[0.58, 0.21, 0.12, 0.07, 0.02]), name="ca")
    thal = pd.Series(np.random.choice(4, batch_size, p=[0.01, 0.06, 0.55, 0.38]), name="thal")

    colunms_data = (age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal)
    return pd.concat(colunms_data, axis=1)


def generate_dataset(
        batch_size: int,
        model: ClassifierModel,
        random_state: int = None
) -> pd.DataFrame:
    """
    Synthetic dataset generator.
    Parameters
    ----------
    batch_size: int
        Batch size.
    model:
        Model to generate target.
    random_state: int
        Random state seed.
    Returns
    -------
        Pandas dataframe.
    """

    df = generate_features(batch_size, random_state)
    df["target"] = model.predict(df)
    return df
