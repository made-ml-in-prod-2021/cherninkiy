from typing import Dict
import logging.config
import joblib
import hydra
import numpy as np

from src.data.utils import read_dataset, split_dataset
from src.features.build_features import FeatureBuilder, TargetBuilder
from src.models.predict_model import make_preds, eval_model
from src.entities.pipeline_params import PipelineParams

logger = logging.getLogger("ml_project/predict_pipeline")


@hydra.main(config_path="../conf", config_name="pipeline")
def predict_pipeline(pipeline_params: PipelineParams) -> np.array:

    logger.info(f"Predict pipeline {pipeline_params.model}")
    logger.info(f"Dataset loading ...")

    df = read_dataset(pipeline_params.data.data_path)

    logger.info("Feature building...")

    X = FeatureBuilder(pipeline_params.features).fit_transform(df)

    logger.info("Model loading...")

    model = joblib.load(pipeline_params.model.path)

    logger.info("Model predicting...")

    preds = make_preds(model, X)

    logger.info("Pipeline is done")

    return preds


if __name__ == "__main__":
    predict_pipeline()