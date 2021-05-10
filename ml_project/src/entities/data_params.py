from dataclasses import dataclass, MISSING


@dataclass()
class DataParams:
    dataset: str = MISSING
    url: str = MISSING
    data_path: str = MISSING
    train_size: float = 0.7