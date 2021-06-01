import numpy as np
import pandas as pd
import logging
from typing import Dict, Union
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

ClassifierModel = Union[LogisticRegression, RandomForestClassifier]
logger = logging.getLogger("ml_project/train_pipeline")


def make_preds(
        model: ClassifierModel,
        features: pd.DataFrame) -> np.ndarray:
    preds = model.predict(features)
    return preds


def eval_model(preds: np.array, target: np.array) -> Dict[str, float]:
    return {
        "roc_auc_score": roc_auc_score(preds, target),
        "accuracy_score": accuracy_score(preds, target),
        "f1_score": f1_score(preds, target),
    }