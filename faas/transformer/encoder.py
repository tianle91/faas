from __future__ import annotations

from typing import Dict, List, Optional

import pyspark.sql.functions as F
from pyspark.sql import DataFrame
from pyspark.sql.types import LongType, NumericType, StringType

from faas.transformer.base import BaseTransformer

from .utils import (clean_string, validate_categorical_types,
                    validate_numeric_types)


def get_distinct_values(df: DataFrame, column: str) -> set:
    if isinstance(df.schema[column].dataType, NumericType):
        raise TypeError(
            f'Column: {column} '
            f'dataType: {df.schema[column].dataType} is a NumericType, '
            'which cannot be used for encoding.'
        )
    DISTINCT_VAL_COL = '__DISTINCT_VAL__'
    distinct_rows = (
        df
        .select(F.col(column).alias(DISTINCT_VAL_COL))
        .distinct()
        .orderBy(DISTINCT_VAL_COL)
        .collect()
    )
    return {row[DISTINCT_VAL_COL] for row in distinct_rows}


class OrdinalEncoder(BaseTransformer):
    def __init__(self, categorical_column: str) -> None:
        self.categorical_column = categorical_column
        self.distincts: list = []

    @property
    def num_classes(self):
        self.check_is_fitted()
        return len(self.distincts)

    @property
    def is_fitted(self):
        return len(self.distincts) > 0

    def check_is_fitted(self):
        if not self.is_fitted:
            raise ValueError('Not fitted yet.')

    def check_is_not_fitted(self):
        if self.is_fitted:
            raise ValueError('Already fitted.')

    @property
    def input_columns(self) -> List[str]:
        return [self.categorical_column]

    @property
    def feature_column(self) -> str:
        return f'OrdinalEncoder_{self.categorical_column}'

    @property
    def feature_columns(self) -> str:
        return [self.feature_column]

    def validate(self, df: DataFrame, is_inverse=False):
        if is_inverse:
            self.check_is_fitted()
            validate_numeric_types(df, cols=self.feature_columns)
        else:
            validate_categorical_types(df, cols=[self.categorical_column])

    def fit(self, df: DataFrame) -> OrdinalEncoder:
        self.validate(df)
        for v in get_distinct_values(df, self.categorical_column):
            if v not in self.distincts:
                self.distincts.append(v)
        return self

    def transform(self, df: DataFrame) -> DataFrame:
        self.validate(df)
        self.check_is_fitted()
        # create udf to do the mapping because joining requires access to a spark session
        mapping = {k: i for i, k in enumerate(self.distincts)}
        udf = F.udf(lambda v: mapping.get(v, None), LongType())
        return df.withColumn(
            self.feature_column,
            udf(F.col(self.categorical_column))
        )

    def inverse_transform(self, df: DataFrame) -> DataFrame:
        self.validate(df, is_inverse=True)
        # create udf to do the mapping because joining requires access to a spark session
        mapping = {i: k for i, k in enumerate(self.distincts)}
        udf = F.udf(lambda i: mapping.get(i, None), StringType())
        return df.withColumn(
            self.categorical_column,
            udf(F.col(self.feature_column))
        )


def array_to_value_mapping(arr: Optional[List[int]], distincts: List[str]) -> Optional[str]:
    if arr is not None:
        if len(arr) != len(distincts):
            raise ValueError(f'{len(arr):} != {len(distincts):}')
        if any([v < 0 for v in arr]):
            raise ValueError(f'All values in {arr:} should be non-negative.')
        if sum(arr) != 1:
            raise ValueError(f'There should only be a single 1 in {arr:}')
        return distincts[arr.index(1)]


class OneHotEncoder(OrdinalEncoder):
    def __init__(self, categorical_column: str) -> None:
        self.categorical_column = categorical_column
        self.distincts: list = []

    @property
    def distinct_to_column_name_mapping(self) -> Dict[str, object]:
        self.check_is_fitted()
        return {
            v: f'OneHotEncoder_{self.categorical_column}_is_{clean_string(v)}'
            for v in self.distincts
        }

    @property
    def feature_columns(self) -> str:
        return list(self.distinct_to_column_name_mapping.values())

    def fit(self, df: DataFrame) -> OrdinalEncoder:
        self.check_is_not_fitted()
        return super().fit(df)

    def transform(self, df: DataFrame) -> DataFrame:
        self.validate(df)
        self.check_is_fitted()
        for v in self.distincts:
            df = df.withColumn(
                self.distinct_to_column_name_mapping[v],
                F.when(F.col(self.categorical_column) == v, F.lit(1)).otherwise(0)
            )
        return df

    def inverse_transform(self, df: DataFrame) -> DataFrame:
        self.validate(df, is_inverse=True)
        udf = F.udf(
            lambda arr: array_to_value_mapping(arr, distincts=self.distincts),
            StringType()
        )
        ONE_HOT_ARRAY_COL = '__ONE_HOT_ARRAY__'
        return (
            df
            .withColumn(
                ONE_HOT_ARRAY_COL,
                F.array(*[F.col(c) for c in self.distinct_to_column_name_mapping.values()])
            )
            .withColumn(self.categorical_column, udf(F.col(ONE_HOT_ARRAY_COL)))
            .drop(ONE_HOT_ARRAY_COL)
        )
