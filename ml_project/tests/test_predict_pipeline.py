import os
import numpy as np
from typing import NoReturn

from src.predict_pipeline import predict_pipeline
from src.entities.pipeline_params import PipelineParams


def test_predict_pipeline(pipeline_params: PipelineParams) -> NoReturn:

    preds = predict_pipeline(pipeline_params)
    assert os.path.exists(pipeline_params.model.path)
    assert np.all(np.isclose(preds, 1.0) | np.isclose(preds, 0.0))