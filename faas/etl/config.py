from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class TargetConfig:
    column: str
    is_categorical: bool = False
    log_transform: bool = False
    categorical_normalization_column: Optional[str] = None
    numerical_normalization_column: Optional[str] = None


@dataclass
class WeightConfig:
    date_column: Optional[str] = None
    annual_decay_rate: float = .1
    group_columns: Optional[List[str]] = None
