import os
import joblib
import logging
import uvicorn
import hydra
import pandas as pd
from typing import Optional, List, Union
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ValidationError

from src.entities.app_params import RequestData, ResponseData
from src.features.build_features import FeatureBuilder
from src.entities.pipeline_params import PipelineParams

ClassifierModel = Union[LogisticRegression, RandomForestClassifier]


logger = logging.getLogger(__name__)
model: Optional[ClassifierModel] = None
transformer: Optional[FeatureBuilder] = None
app = FastAPI()


@hydra.main(config_path="../conf", config_name="pipeline")
def load_model(pipeline_params: PipelineParams):
    model_path = os.getenv("PATH_TO_MODEL", default="models/logreg.pkl")
    if model_path is None:
        err = f"PATH_TO_MODEL {model_path} is None"
        logger.error(err)
        raise RuntimeError(err)

    logger.info("Model loading...")
    model = joblib.load(pipeline_params.model.path)
    logger.info("Model predicting...")

    transformer = FeatureBuilder(pipeline_params.features)


def make_preds(
    pipeline_params: PipelineParams,
    data: List[RequestData], pipeline: Pipeline) -> List[ResponseData]:
    df = pd.DataFrame(x.__dict__ for x in data)
    X = transformer.fit_transform(df)
    ids = [int(x) for x in data.id]
    predicts = pipeline.predict(data.drop("id", axis=1))

    return [
        ResponseData(id=id_, target=int(target_))
        for id_, target_ in zip(ids, predicts)
    ]


@app.get("/")
def main():
    return "This is the entry point of our predictor."


@app.on_event("startup")
def startup():
    load_model()


@app.get("/status")
def status() -> bool:
    return f"Pipeline is ready: {model is not None}."


@app.api_route("/predict", response_model=List[ResponseData], methods=["GET", "POST"])
def predict(request: List[RequestData]):
    return make_preds(request, model)


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=os.getenv("PORT", 8000))