from typing import List

from pyspark.sql import DataFrame
from pyspark.sql.types import DataType, NumericType, StringType, TimestampType


def validate_types(df: DataFrame, cols: List[str], expected_dtype: DataType):
    for c in cols:
        dtype = df.schema[c].dataType
        if not isinstance(dtype, expected_dtype):
            raise TypeError(f'Column {c} is {dtype} but is expected to be {expected_dtype}.')


def validate_numeric_types(df: DataFrame, cols: List[str]):
    return validate_types(df=df, cols=cols, expected_dtype=NumericType)


def validate_categorical_types(df: DataFrame, cols: List[str]):
    return validate_types(df=df, cols=cols, expected_dtype=StringType)


def validate_timestamp_types(df: DataFrame, cols: List[str]):
    return validate_types(df=df, cols=cols, expected_dtype=TimestampType)


def clean_string(s: str) -> str:
    return ''.join([c if c.isalnum() else '_' for c in s])
