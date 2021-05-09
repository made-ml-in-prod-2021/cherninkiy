import pytest
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from typing import NoReturn, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.validation import check_is_fitted

from src.models.train_model import build_model, train_model
from src.models.predict_model import make_preds, eval_model


def test_logistic_regression_preds(
        params: OmegaConf,
        train_data: Tuple[np.array, np.array],
        test_data: Tuple[np.array, np.array]
    ) -> NoReturn:

    model = train_model(train_data[0], train_data[1], params.logreg)

    check_is_fitted(model)
    assert isinstance(model, LogisticRegression)

    preds = make_preds(model, test_data[0])
    score = eval_model(preds, test_data[1])



def test_random_forest_preds(
        params: OmegaConf,
        train_data: Tuple[np.array, np.array],
        test_data: Tuple[np.array, np.array]
    ) -> NoReturn:

    model = train_model(train_data[0], train_data[1], params.ranfor)

    check_is_fitted(model)
    assert isinstance(model, RandomForestClassifier)

    preds = make_preds(model, test_data[0])
    score = eval_model(preds, test_data[1])