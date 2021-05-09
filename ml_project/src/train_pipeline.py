import os
import logging.config
import numpy as np
import pandas as pd
from typing import Union, Dict
import joblib
import hydra
from omegaconf import OmegaConf
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

from src.data.utils import read_dataset, split_dataset
from src.features.build_features import FeatureBuilder, TargetBuilder
from src.models.train_model import train_model
from src.models.predict_model import make_preds, eval_model


logger = logging.getLogger("ml_project/train_pipeline")

@hydra.main(config_path="../conf", config_name="pipeline")
def train_pipeline(pipeline_params: OmegaConf) -> Dict[str, float]:

    logger.info(f"Train pipeline {pipeline_params.model}")

    logger.info(f"Dataset loading ...")

    df = read_dataset(pipeline_params.data.data_path)

    train_size = pipeline_params.data.train_size
    train, test = split_dataset(df, train_size)

    logger.info("Feature building...")

    feature_builder = FeatureBuilder(pipeline_params.feats)
    target_builder = TargetBuilder(pipeline_params.feats)

    X_train = feature_builder.fit_transform(train)
    y_train = target_builder.fit_transform(train)

    X_test = feature_builder.fit_transform(train)
    y_test = target_builder.fit_transform(train)

    logger.info("Model training...")

    model = train_model(X_train, y_train, pipeline_params.model)

    logger.info("Model evaluationg...")

    preds = make_preds(model, X_test)
    metrics = eval_model(preds, y_test)

    logger.info(f"Metrics: {metrics}")
    logger.info("Pipeline is done")

    joblib.dump(model, pipeline_params.model.path)

    logger.info("Model saved into {pipeline_params.model.path}")

    return metrics


if __name__ == "__main__":
    train_pipeline()