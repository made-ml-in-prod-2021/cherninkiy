import os
import joblib
import logging
import uvicorn
import pandas as pd
from typing import Optional, List, NoReturn
from sklearn.pipeline import Pipeline
from fastapi import FastAPI, HTTPException

from src.entities.app_params import RequestData, ResponseData


PREDICTOR_HOST = os.environ.get("HOST", default="0.0.0.0")
PREDICTOR_PORT = os.environ.get("PORT", default=8080)

logger = logging.getLogger(__name__)
model: Optional[Pipeline] = None
app = FastAPI()


def load_model() -> NoReturn:
    global model

    classifier_path = os.getenv(
        "CLASSIFIER_PATH",
        default="models/logreg.pkl")
    if classifier_path is None:
        err = f"variable not set: CLASSIFIER_PATH"
        logger.error(err)
        raise RuntimeError(err)

    transformer_path = os.getenv(
        "TRANSFORMER_PATH",
        default="models/transformer.pkl")
    if transformer_path is None:
        err = f"variable not set: PATH_TO_TRANSFORMER"
        logger.error(err)
        raise RuntimeError(err)

    logger.info(f"Classifier found: {classifier_path}")

    classifier = joblib.load(classifier_path)

    logger.info("Classifier loaded")

    logger.info(f"Transformer found: {transformer_path}")

    transformer = joblib.load(transformer_path)

    logger.info("Transformer loaded")

    model = Pipeline([
        ('transformer', transformer),
        ('classifier', classifier)
    ])

    logger.info("Pipeline created")



def make_preds(
    data: List[RequestData], pipeline: Pipeline) -> List[ResponseData]:
    df = pd.DataFrame(x.__dict__ for x in data)

    # correct features order
    columns = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
    ]

    df = df[columns]
    df['target'] = None

    ids = list(range(len(data)))
    predicts = model.predict(df)

    return [
        ResponseData(id=id_, target=int(target_))
        for id_, target_ in zip(ids, predicts)
    ]


@app.get("/")
def main():
    return "This is the entry point of our predictor."


@app.on_event("startup")
def startup():
    logger.info("Starting service...")
    load_model()


@app.get("/status")
def status() -> bool:
    if model is not None:
        return "Predictor is ready"
    return "Predictor not ready"


@app.api_route("/predict", response_model=List[ResponseData], methods=["GET", "POST"])
def predict(request: List[RequestData]):
    global model
    return make_preds(request, model)


if __name__ == "__main__":
    uvicorn.run("app:app", host=PREDICTOR_HOST, port=os.getenv("PORT", PREDICTOR_PORT))