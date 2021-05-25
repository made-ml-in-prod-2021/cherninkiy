from typing import Dict
import logging
import joblib
import hydra

from src.data.utils import read_dataset, split_dataset
from src.features.build_features import FeatureBuilder, TargetBuilder
from src.models.train_model import train_model
from src.models.predict_model import make_preds, eval_model
from src.entities.pipeline_params import PipelineParams

logger = logging.getLogger("ml_project/train_pipeline")


@hydra.main(config_path="../conf", config_name="pipeline")
def train_pipeline(pipeline_params: PipelineParams) -> Dict[str, float]:

    logger.info(f"Train pipeline {pipeline_params.model}")
    logger.info(f"Dataset loading ...")

    df = read_dataset(pipeline_params.data.data_path)

    train_size = pipeline_params.data.train_size
    train, test = split_dataset(df, train_size)

    logger.info("Feature building...")

    feature_builder = FeatureBuilder(pipeline_params.features)
    target_builder = TargetBuilder(pipeline_params.features)

    X_train = feature_builder.fit_transform(train)
    y_train = target_builder.fit_transform(train)

    X_test = feature_builder.fit_transform(train)
    y_test = target_builder.fit_transform(train)

    logger.info("Model training...")

    model = train_model(X_train, y_train, pipeline_params.model)

    logger.info("Model evaluating...")

    preds = make_preds(model, X_test)
    metrics = eval_model(preds, y_test)

    logger.info(f"Metrics: {metrics}")
    logger.info("Pipeline is done")

    model_path = pipeline_params.model.path
    joblib.dump(model, model_path)

    logger.info(f"Model saved into {model_path}")

    transformer_path = pipeline_params.features.transformer_path
    joblib.dump(feature_builder, transformer_path)

    logger.info(f"Transformer saved into {transformer_path}")

    return metrics


if __name__ == "__main__":
    train_pipeline()