import numpy as np
import pandas as pd
from typing import Union
from omegaconf import OmegaConf
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


ClassifierModel = Union[LogisticRegression, RandomForestClassifier]

def build_model(
        params: OmegaConf
    ) -> pd.DataFrame:
    """
    Model builder.
    params: OmegaConf
        Model params.
    """
    if params.model == "LogisticRegression":
        return LogisticRegression(**params.kwargs)
    elif params.model == "RandomForestClassifier":
        return RandomForestClassifier(**params.kwargs)
    else:
        raise ValueError(f"Unknown model {params.model}")


def train_model(
            features: np.array,
            target: np.array,
            params: OmegaConf,
    ) -> ClassifierModel:
    """
    Model training.
    features: np.array
    target: np.array
    params: OmegaConf
        Model params.
    """
    model = build_model(params)
    model.fit(features, target)

    return model