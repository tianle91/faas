from __future__ import annotations

from typing import List

import pyspark.sql.functions as F
from pyspark.sql import DataFrame
from pyspark.sql.types import LongType, NumericType, StringType

from faas.utils_dataframe import (validate_categorical_types,
                                  validate_numeric_types)


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


class OrdinalEncoderSingle:
    def __init__(self, categorical_column: str) -> None:
        self.categorical_column = categorical_column
        self.distincts: list = []
        self.column_type = None

    def validate(self, df: DataFrame):
        validate_categorical_types(df, cols=[self.categorical_column])

    def fit(self, df: DataFrame) -> OrdinalEncoderSingle:
        self.validate(df)
        for v in get_distinct_values(df, self.categorical_column):
            if v not in self.distincts:
                self.distincts.append(v)
        return self

    def transform(self, df: DataFrame) -> DataFrame:
        self.validate(df)
        mapping = {k: i for i, k in enumerate(self.distincts)}
        udf = F.udf(lambda v: mapping.get(v, None), LongType())
        return df.withColumn(self.categorical_column, udf(F.col(self.categorical_column)))

    def inverse_transform(self, df: DataFrame) -> DataFrame:
        validate_numeric_types(df, cols=[self.categorical_column])
        inverse_mapping = {i: k for i, k in enumerate(self.distincts)}
        udf = F.udf(lambda v: inverse_mapping.get(v, None), StringType())
        return df.withColumn(self.categorical_column, udf(F.col(self.categorical_column)))


class OrdinalEncoder:
    def __init__(self, categorical_columns: List[str]):
        self.encoder = {c: OrdinalEncoderSingle(categorical_column=c) for c in categorical_columns}

    def fit(self, df: DataFrame) -> OrdinalEncoder:
        for enc in self.encoder.values():
            enc.fit(df)
        return self

    def transform(self, df: DataFrame) -> DataFrame:
        for enc in self.encoder.values():
            df = enc.transform(df)
        return df

    def inverse_transform(self, df: DataFrame) -> DataFrame:
        for enc in self.encoder.values():
            df = enc.inverse_transform(df)
        return df
