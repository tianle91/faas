from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ETLConfig:
    x_categorical_columns: List[str]
    x_numeric_features: List[str]
    target_column: str
    target_log_transform: bool = False
    target_normalize_by_categorical: Optional[str] = None
    target_normalize_by_numerical: Optional[str] = None
    date_column: Optional[str] = None
    weight_group_columns: Optional[List[str]] = None
