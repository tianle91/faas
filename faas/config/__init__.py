from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from pyspark.sql import DataFrame
from pyspark.sql.types import NumericType


def equality_or_both_none(a: Optional[object], b: Optional[object]) -> bool:
    if a is None and b is None:
        return True
    else:
        return a == b


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

    def is_same_problem(self, other: Config) -> bool:
        return all([
            equality_or_both_none(getattr(self, k), getattr(other, k))
            for k in self.__dict__ if 'feature' not in k
        ])

    def validate_df(self, df: DataFrame, prediction: bool = False):
        if not prediction:
            if self.target not in df.columns:
                raise ValueError(f'Target: {self.target} not in df.columns')
            target_dtype = df.schema[self.target].dataType
            if not self.target_is_categorical and not isinstance(target_dtype, NumericType):
                raise ValueError(f'Expected numeric target but received {target_dtype} instead.')

        if self.date_column is not None and self.date_column not in df.columns:
            raise ValueError(f'Date column: {self.date_column} not in df.columns')
        if self.has_spatial_columns and not(
            self.latitude_column in df.columns
            and self.longitude_column in df.columns
        ):
            raise ValueError(
                f'Spatial columns: {self.latitude_column}, {self.longitude_column} '
                'are not in df.columns'
            )
        if self.group_columns is not None:
            missing_group_columns = [c for c in self.group_columns if c not in df.columns]
            if len(missing_group_columns) > 0:
                raise ValueError(
                    f'Group columns: {missing_group_columns} are not in df.columns')
        if self.feature_columns is not None:
            missing_feature_columns = [c for c in self.feature_columns if c not in df.columns]
            if len(missing_feature_columns) > 0:
                raise ValueError(
                    f'Feature columns: {missing_feature_columns} are not in df.columns')

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
