import logging
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.exceptions import NotFittedError

from src.entities.feature_params import FeatureParams

logger = logging.getLogger("ml_project/train_pipeline")


class FeatureBuilder(BaseEstimator, TransformerMixin):
    """ Class that incapsulates feature building logic. """
    def __init__(self, params: FeatureParams):
        self.params = params
        self.fitted = False
        self.pipeline = None


    def check_is_fitted(self):
        if not self.fitted:
            raise NotFittedError("FeatureBuilder not fitted")


    def build_categorical_pipeline(self) -> Pipeline:
        return Pipeline([
            ("imputer", SimpleImputer(missing_values=np.nan, strategy="most_frequent")),
        ])


    def build_numerical_pipeline(self) -> Pipeline:
        return Pipeline([
            ("imputer", SimpleImputer(missing_values=np.nan, strategy="mean")),
        ])


    def build_numerical_scaled_pipeline(self) -> Pipeline:
        return Pipeline([
            ("imputer", SimpleImputer(missing_values=np.nan, strategy="mean")),
            ("scaler", StandardScaler()),
        ])


    def fit(self, X: pd.DataFrame):
        categorical_feats = list(self.params.categorical_feats)
        numerical_feats = list(self.params.numerical_feats)
        if self.params.normalize_numerical:
            self.pipeline = ColumnTransformer(
                [
                    ("categorical_pipeline", self.build_categorical_pipeline(), categorical_feats),
                    ("numerical_pipeline",  self.build_numerical_pipeline(), numerical_feats),
                ]
            )

            logger.info("FeatureBuilder pipline (normilized) fitting")

        else:
            self.pipeline = ColumnTransformer(
                [
                    ("categorical_pipeline", self.build_categorical_pipeline(), categorical_feats),
                    ("numerical_scaled_pipeline",  self.build_numerical_scaled_pipeline(), numerical_feats),
                ]
            )

            logger.info("FeatureBuilder pipline fitting")

        self.pipeline.fit(X)
        self.fitted = True
        return self


    def transform(self, X: pd.DataFrame) -> np.array:
        self.check_is_fitted()
        if X.empty:
            return pd.DataFrame([])
        res = self.pipeline.transform(X)

        logger.info("FeatureBuilder pipline transforming")

        return res


    def fit_transform(self, X: pd.DataFrame) -> np.array:
        self.fit(X)
        return self.transform(X)


class TargetBuilder(BaseEstimator, TransformerMixin):
    """ Class that incapsulates target building logic. """
    def __init__(self, params: FeatureParams):
        self.target_col = params.target_col


    def fit(self, X: pd.DataFrame):
        return self


    def transform(self, X: pd.DataFrame) -> np.array:
        if X.empty:
            return pd.Series([]).values
        return X[self.target_col].values


    def fit_transform(self, X: pd.DataFrame) -> np.array:
        self.fit(X)
        return self.transform(X)