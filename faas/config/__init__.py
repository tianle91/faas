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
        if self.feature_columns is None:
            raise ValueError('Feature columns cannot be None')

    @property
    def used_columns_prediction(self) -> List[str]:
        out = self.feature_columns
        if self.date_column is not None:
            out.append(self.date_column)
        if self.space_columns is not None:
            out += list(self.space_columns)
        if self.group_columns is not None:
            out += self.group_columns
        return out
