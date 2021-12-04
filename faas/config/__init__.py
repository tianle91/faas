from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Config:
    target: str = 'target'
    target_is_categorical: bool = False
    date_column: Optional[str] = None
    date_column_format: str = 'yyyy-MM-dd'
    latitude_column: Optional[str] = None
    longitude_column: Optional[str] = None
    group_columns: Optional[List[str]] = None
    feature_columns: Optional[List[str]] = None

    def validate(self):
        if self.feature_columns is None:
            raise ValueError('Feature columns cannot be None')

    @property
    def has_spatial_columns(self):
        return self.latitude_column is not None and self.longitude_column is not None

    @property
    def used_columns_prediction(self) -> List[str]:
        out = self.feature_columns
        if self.date_column is not None:
            out.append(self.date_column)
        if self.latitude_column is not None and self.longitude_column:
            out += [self.latitude_column, self.longitude_column]
        if self.group_columns is not None:
            out += self.group_columns
        return out
