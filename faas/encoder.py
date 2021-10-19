import pyspark.sql.functions as F
from pyspark.sql import DataFrame
from pyspark.sql.types import LongType, NumericType


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


class OrdinalEncoderSingleSpark:
    def __init__(self, column: str) -> None:
        self.column = column
        self.distincts: list = []

    def fit(self, df: DataFrame):
        new_values = get_distinct_values(df, self.column)
        new_values = [v for v in new_values if v not in self.distincts]
        self.distincts += new_values

    def transform(self, df: DataFrame) -> DataFrame:
        mapping = {k: i for i, k in enumerate(self.distincts)}
        return df.withColumn(
            self.column,
            F.udf(
                lambda v: mapping.get(v, None),
                LongType()
            )(F.col(self.column))
        )
