from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class Config:
    target: str = 'target'
    target_is_categorical: bool = False
    date_column: Optional[str] = None
    date_column_format: str = 'yyyy-MM-dd'
    space_columns: Optional[Tuple[str, str]] = None
    group_columns: Optional[List[str]] = None
    feature_columns: Optional[List[str]] = None

    def validate(self):
        if self.target_is_categorical:
            raise ValueError('Categorical column not yet supported')
