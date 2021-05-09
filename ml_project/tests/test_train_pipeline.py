import os
from omegaconf import OmegaConf
from typing import NoReturn
from src.train_pipeline import train_pipeline


def test_train_pipeline(pipeline_params: OmegaConf) -> NoReturn:

    metrics = train_pipeline(pipeline_params)

    assert os.path.exists(pipeline_params.model.path)
    assert 0 < metrics["roc_auc_score"] <= 1
    assert 0 < metrics["accuracy_score"] <= 1
    assert 0 < metrics["f1_score"] <= 1