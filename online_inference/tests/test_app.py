import pytest
import joblib
import hydra
from fastapi.testclient import TestClient

from app import app
from src.data.utils import read_dataset
from src.entities.pipeline_params import PipelineParams
from src.features.build_features import FeatureBuilder
from src.models.predict_model import make_preds


@pytest.fixture
def client():
    with TestClient(app) as client:
        yield client


@pytest.fixture
@hydra.main(config_path="../conf", config_name="pipeline")
def test_data(pipeline_params: PipelineParams):
    df = read_dataset(pipeline_params.data.data_path)
    X = FeatureBuilder(pipeline_params.features).fit_transform(df)
    model = model=joblib.load(pipeline_params.model.path)
    preds = make_preds(model, X)
    return preds


def test_startup(client):
    resp = client.get("/status")

    assert resp.status_code == 200
    assert resp.json() is True


@pytest.mark.parametrize(
    ["feature", "value", "status_code"],
    [
        pytest.param("age", 180, 400),
        pytest.param("sex", 2, 400),
        pytest.param("ca", 7, 400),
        pytest.param("trestbps", 250, 400),
    ]
)
def test_broken_data(feature, value, status_code, test_data, client):
    broken_data = test_data[0].copy()
    broken_data[feature] = value
    resp = client.get("/predict", json=broken_data)

    assert resp.status_code == status_code


def test_predict_target(test_data, client):
    expected_status_code = 200
    expected_value = {"target": test_data[1]}
    resp = client.get("/predict", json=test_data[0])

    assert resp.status_code == expected_status_code
    assert resp.json() == expected_value