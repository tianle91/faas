from typing import List

from pyspark.sql import DataFrame
from pyspark.sql.types import NumericType, StringType, TimestampType


def get_non_numeric_columns(df: DataFrame) -> List[str]:
    return [c for c in df.columns if not isinstance(df.schema[c].dataType, NumericType)]


def validate_numeric_types(df: DataFrame, cols: List[str]):
    for c in cols:
        dtype = df.schema[c].dataType
        if not isinstance(dtype, NumericType):
            raise TypeError(f'Column {c} is {dtype} but is expected to be numeric.')


def validate_categorical_types(df: DataFrame, cols: List[str]):
    for c in cols:
        dtype = df.schema[c].dataType
        if not isinstance(dtype, StringType):
            raise TypeError(f'Column {c} is {dtype} but is expected to be string.')


def validate_timestamp_types(df: DataFrame, cols: List[str]):
    for c in cols:
        dtype = df.schema[c].dataType
        if not isinstance(dtype, TimestampType):
            raise TypeError(f'Column {c} is {dtype} but is expected to be timestamp.')
