import joblib
import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from ..src.data.generate import generate_features, generate_dataset


def test_generate_features(params: OmegaConf):

    batch_size = params.synthetic.batch_size
    random_state = params.synthetic.random_state
    df = generate_features(batch_size, random_state)

    features = sorted(("age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
               "thalach", "exang", "oldpeak", "slope", "ca", "thal"))

    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == batch_size
    assert df.shape[1] == len(features)
    assert all([c1 == c2 for c1, c2 in zip(features, sorted(df.columns))])


def test_generate_dataset(params: OmegaConf):

    batch_size = params.synthetic.batch_size
    random_state = params.synthetic.random_state

    model_name = params.synthetic.used_model
    model_path = params[model_name].path
    model = joblib.load(model_path)

    df = generate_dataset(batch_size, model, random_state)

    columns = sorted(("age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
               "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"))

    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == batch_size
    assert df.shape[1] == len(columns)
    assert all([c1 == c2 for c1, c2 in zip(columns, sorted(df.columns))])

    assert np.all(np.isclose(df["target"], 1.0) | np.isclose(df["target"], 0.0))
