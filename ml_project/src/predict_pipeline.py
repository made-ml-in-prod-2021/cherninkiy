import os
import logging.config
import numpy as np
import pandas as pd
import joblib
import hydra
from typing import Union, Dict
from omegaconf import OmegaConf
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

from src.data.utils import read_dataset, split_dataset
from src.features.build_features import FeatureBuilder, TargetBuilder
from src.models.train_model import train_model
from src.models.predict_model import make_preds, eval_model
from src.entities.model_params import ModelParams

logger = logging.getLogger("ml_project/predict_pipeline")

@hydra.main(config_path="../conf", config_name="pipeline")
def predict_pipeline(pipeline_params: ModelParams) -> Dict[str, float]:

    logger.info(f"Predict pipeline {pipeline_params.model}")

    logger.info(f"Dataset loading ...")

    df = read_dataset(pipeline_params.data.data_path)

    logger.info("Feature building...")

    X = FeatureBuilder(pipeline_params.feats).fit_transform(df)

    logger.info("Model loading...")

    model = model=joblib.load(pipeline_params.model.path)

    logger.info("Model predicting...")

    preds = make_preds(model, X)

    logger.info("Pipeline is done")

    return preds


if __name__ == "__main__":
    train_pipeline()