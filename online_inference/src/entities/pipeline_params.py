from dataclasses import dataclass, MISSING

from src.entities.data_params import DataParams
from src.entities.feature_params import FeatureParams
from src.entities.model_params import ModelParams


@dataclass()
class PipelineParams:
    data: DataParams = MISSING
    features: FeatureParams = MISSING
    model: ModelParams = MISSING