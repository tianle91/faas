from typing import List

import pyspark.sql.functions as F
from pyspark.sql import DataFrame
from pyspark.sql.types import DateType, DoubleType, NumericType, StringType


def is_numeric(df: DataFrame, column: str) -> bool:
    return isinstance(df.schema[column].dataType, NumericType)


def check_columns_are_desired_type(
    columns: List[str], dtype: DataType, df: DataFrame
) -> Tuple[bool, List[str]]:
    all_passed = True
    messages = []
    for c in columns:
        actual = df.schema[c].dataType
        if not isinstance(actual, dtype):
            all_passed = False
            messages.append(
                f'Expected column: {c} to be {dtype} but received {actual} instead.')
    return all_passed, messages


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


def validate_date_types(df: DataFrame, cols: List[str]):
    for c in cols:
        dtype = df.schema[c].dataType
        if not isinstance(dtype, DateType):
            raise TypeError(f'Column {c} is {dtype} but is expected to be timestamp.')


ROW_ID_COL = '__ROW_ID__'


class JoinableByRowID:
    """Adds a row id column and makes a dataframe joinable by row id."""

    def __init__(self, df: DataFrame):
        self.df = (
            df
            .withColumn(ROW_ID_COL, F.monotonically_increasing_id())
            .orderBy(ROW_ID_COL)
        )

    def join_by_row_id(self, x: List[float], column: str) -> DataFrame:
        mapping = {i: float(val) for i, val in enumerate(x)}
        udf = F.udf(lambda i: mapping[i], DoubleType())
        return self.df.withColumn(column, udf(F.col(ROW_ID_COL)))
