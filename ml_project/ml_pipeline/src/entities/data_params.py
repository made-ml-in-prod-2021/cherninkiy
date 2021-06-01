from typing import Optional
from dataclasses import dataclass, MISSING


@dataclass()
class DataParams:
    dataset: str = MISSING
    url: str = MISSING
    data_path: str = MISSING
    train_size: Optional[float] = 0.7