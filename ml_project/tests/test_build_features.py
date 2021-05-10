import pytest
import numpy as np
import pandas as pd

from src.data.utils import read_dataset
from src.features.build_features import FeatureBuilder, TargetBuilder
from src.entities.feature_params import FeatureParams

def test_feature_builder(params: FeatureParams):

    feature_builder = FeatureBuilder(params.features)

    df = read_dataset(params.data.data_path)
    features = feature_builder.fit_transform(df)

    assert isinstance(features, np.ndarray)


def test_target_builder(params: FeatureParams):

    target_builder = TargetBuilder(params.features)

    df = read_dataset(params.data.data_path)
    target = target_builder.fit_transform(df)

    assert isinstance(target, np.ndarray)
    assert (df.shape[0],) == target.shape
