import pytest
from fastapi.testclient import TestClient

from app import app


@pytest.fixture
def client():
    with TestClient(app) as client:
        yield client

@pytest.fixture
def test_data():
    data = {
        'age': 57, 'sex': 0, 'cp': 1, 'trestbps': 130, 'chol': 236, 'fbs': 0, 'restecg': 0,
        'thalach': 174, 'exang': 0, 'oldpeak': 0.0, 'slope': 1, 'ca': 1, 'thal': 2, 'target': 1
    }

    target = data['target']
    data.pop('target')
    return data, target


def test_startup(client):
    resp = client.get("/status")

    assert resp.status_code == 200
    assert resp.json() == f"Predictor is ready"


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
    broken_data = [test_data[0].copy()]
    broken_data[0][feature] = value
    resp = client.get("/predict", json=broken_data)

    assert resp.status_code == status_code


def test_predict_target(test_data, client):
    expected_status_code = 200
    expected_value = {"target": test_data[1]}
    resp = client.get("/predict", json=[test_data[0]])

    assert resp.status_code == expected_status_code
    assert resp.json() == [expected_value]