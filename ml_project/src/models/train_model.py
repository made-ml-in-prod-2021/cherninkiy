import logging
import numpy as np
from typing import Union
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from src.entities.model_params import ModelParams

ClassifierModel = Union[LogisticRegression, RandomForestClassifier]
logger = logging.getLogger("ml_project/train_pipeline")


def build_model(params: ModelParams) -> ClassifierModel:
    """
    Model builder.
    params: ModelParams
        Model params.
    """
    if params.model == "LogisticRegression":
        logger.info("LogisticRegression model configured")
        return LogisticRegression(**params.kwargs)
    elif params.model == "RandomForestClassifier":
        logger.info("RandomForestClassifier model configured")
        return RandomForestClassifier(**params.kwargs)
    else:
        raise ValueError(f"Unknown model {params.model}")


def train_model(
            features: np.array,
            target: np.array,
            params: ModelParams,
    ) -> ClassifierModel:
    """
    Model training.
    features: np.array
    target: np.array
    params: ModelParams
        Model params.
    """
    model = build_model(params)
    model.fit(features, target)
    return model