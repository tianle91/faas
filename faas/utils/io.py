from io import BytesIO
from typing import List, Optional

import pyspark.sql.functions as F
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import DoubleType


def dump_file_to_location(file: BytesIO, p: str):
    with open(p, 'wb') as f:
        f.write(file.read())


def load_csv_with_types(
    spark: SparkSession,
    p: str,
    numeric_columns: Optional[List[str]] = None,
    categorical_columns: Optional[List[str]] = None,
    date_columns: Optional[List[str]] = None,
    date_format: str = 'yyyy-MM-dd'
) -> DataFrame:
    df = spark.read.options(header=True, inferSchema=False).csv(p)
    all_cols = []
    if numeric_columns is not None:
        all_cols += numeric_columns
        for c in numeric_columns:
            df = df.withColumn(c, F.col(c).cast(DoubleType()))
    if categorical_columns is not None:
        all_cols += categorical_columns
    if date_columns is not None:
        all_cols += date_columns
        for c in numeric_columns:
            df = df.withColumn(c, F.to_date(F.col(c), format=date_format))
    if len(all_cols) == 0:
        raise ValueError('all_cols has no entries')
    df = df.select(*all_cols)
    return df
