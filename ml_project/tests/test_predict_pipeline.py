import os
from omegaconf import OmegaConf
from typing import NoReturn
from src.predict_pipeline import predict_pipeline


def test_predict_pipeline(pipeline_params: OmegaConf) -> NoReturn:

    preds = predict_pipeline(pipeline_params)

    assert os.path.exists(pipeline_params.model.path)