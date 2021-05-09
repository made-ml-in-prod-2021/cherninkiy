from dataclasses import dataclass, field, MISSING
from typing import Dict, Optional


@dataclass()
class ModelParams:
    model: str = "LogisticRegression"
    path: str = "models/logreg.pkl"
    kwargs: Optional[Dict] = field(default_factory=dict)