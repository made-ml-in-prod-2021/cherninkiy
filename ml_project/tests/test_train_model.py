import numpy as np
from typing import NoReturn, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.validation import check_is_fitted

from src.models.train_model import build_model, train_model
from src.entities.model_params import ModelParams


def test_logistic_regression_build(params: ModelParams) -> NoReturn:

    model = build_model(params.logreg)
    assert isinstance(model, LogisticRegression)


def test_logistic_regression_train(
        params: ModelParams,
        train_data: Tuple[np.array, np.array]
    ) -> NoReturn:

    model = train_model(train_data[0], train_data[1], params.logreg)
    check_is_fitted(model)
    assert isinstance(model, LogisticRegression)


def test_random_forest_build(params: ModelParams) -> NoReturn:

    model = build_model(params.ranfor)
    assert isinstance(model, RandomForestClassifier)


def test_random_forest_train(
        params: ModelParams,
        train_data: Tuple[np.array, np.array]
    ) -> NoReturn:

    model = train_model(train_data[0], train_data[1], params.ranfor)
    check_is_fitted(model)
    assert isinstance(model, RandomForestClassifier)