from dataclasses import dataclass, MISSING

from .data_params import DataParams
from .feature_params import FeatureParams
from .model_params import ModelParams
from .transformer_params import TransformerParams

@dataclass()
class PipelineParams:
    data: DataParams = MISSING
    features: FeatureParams = MISSING
    model: ModelParams = MISSING
    transformer: TransformerParams = MISSING