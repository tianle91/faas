from __future__ import annotations

import pyspark.sql.functions as F
from pyspark.sql import DataFrame
from pyspark.sql.types import LongType, NumericType

from faas.transformer.base import BaseTransformer
from faas.utils_dataframe import validate_categorical_types


def get_distinct_values(df: DataFrame, column: str) -> set:
    if isinstance(df.schema[column].dataType, NumericType):
        raise TypeError(
            f'Column: {column} '
            f'dataType: {df.schema[column].dataType} is a NumericType, '
            'which cannot be used for encoding.'
        )
    distinct_rows = (
        df
        .select(F.col(column).alias('val'))
        .distinct()
        .collect()
    )
    return {row.val for row in distinct_rows}


class OrdinalEncoder(BaseTransformer):
    def __init__(self, categorical_column: str) -> None:
        self.categorical_column = categorical_column
        self.distincts: list = []
        self.column_type = None

    @property
    def feature_column(self) -> str:
        return f'OrdinalEncoder_{self.categorical_column}'

    @property
    def feature_columns(self) -> str:
        return [self.feature_column]

    def validate(self, df: DataFrame):
        validate_categorical_types(df, cols=[self.categorical_column])

    def fit(self, df: DataFrame) -> OrdinalEncoder:
        self.validate(df)
        for v in get_distinct_values(df, self.categorical_column):
            if v not in self.distincts:
                self.distincts.append(v)
        return self

    def transform(self, df: DataFrame) -> DataFrame:
        self.validate(df)
        # create udf to do the mapping because joining requires access to a spark session
        mapping = {k: i for i, k in enumerate(self.distincts)}
        udf = F.udf(lambda v: mapping.get(v, None), LongType())
        return df.withColumn(
            self.feature_column,
            udf(F.col(self.categorical_column))
        )
