import joblib
import numpy as np
from typing import NoReturn, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.validation import check_is_fitted

from ..src.models.train_model import train_model
from ..src.models.predict_model import make_preds, eval_model
from ..src.entities.model_params import ModelParams
from ..src.entities.pipeline_params import PipelineParams


def test_logistic_regression_preds(
        params: ModelParams,
        train_data: Tuple[np.array, np.array],
        test_data: Tuple[np.array, np.array]
    ) -> NoReturn:

    model = train_model(train_data[0], train_data[1], params.logreg)

    check_is_fitted(model)
    assert isinstance(model, LogisticRegression)

    preds = make_preds(model, test_data[0])
    assert preds.shape[0] > 0 and preds.shape[0] == test_data[0].shape[0]
    assert np.all(np.isclose(preds, 1.0) | np.isclose(preds, 0.0))

    metrics = eval_model(preds, test_data[1])
    assert 0 < metrics["roc_auc_score"] <= 1
    assert 0 < metrics["accuracy_score"] <= 1
    assert 0 < metrics["f1_score"] <= 1


def test_random_forest_preds(
        params: ModelParams,
        train_data: Tuple[np.array, np.array],
        test_data: Tuple[np.array, np.array]
    ) -> NoReturn:

    model = train_model(train_data[0], train_data[1], params.ranfor)

    check_is_fitted(model)
    assert isinstance(model, RandomForestClassifier)

    preds = make_preds(model, test_data[0])
    assert preds.shape[0] > 0 and preds.shape[0] == test_data[0].shape[0]
    assert np.all(np.isclose(preds, 1.0) | np.isclose(preds, 0.0))

    metrics = eval_model(preds, test_data[1])
    assert 0 < metrics["roc_auc_score"] <= 1
    assert 0 < metrics["accuracy_score"] <= 1
    assert 0 < metrics["f1_score"] <= 1


def test_synthetic_data_preds(
        pipeline_params: PipelineParams,
        synthetic_data: np.array
):
    model = joblib.load(pipeline_params.model.path)

    check_is_fitted(model)
    assert isinstance(model, LogisticRegression) \
           | isinstance(model, RandomForestClassifier)

    preds = make_preds(model, synthetic_data)
    assert preds.shape[0] > 0 and preds.shape[0] == synthetic_data.shape[0]
    assert np.all(np.isclose(preds, 1.0) | np.isclose(preds, 0.0))