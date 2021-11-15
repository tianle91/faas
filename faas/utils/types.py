from typing import List

from pyspark.sql import DataFrame, SparkSession

from faas.utils.dataframe import get_numeric_columns

DEFAULT_DATE_FORMAT = 'yyyy-MM-dd'


def load_csv(spark: SparkSession, p: str) -> DataFrame:
    df = (
        spark
        .read
        .format('csv')
        .options(header=True, inferSchema=True, dateFormat=DEFAULT_DATE_FORMAT)
        .load(p)
    )
    return df


def numeric_with_few_distincts(df: DataFrame, num_distincts: int = 10) -> List[str]:
    out = []
    for c in get_numeric_columns(df):
        if df.select(c).distinct().count() < num_distincts:
            out.append(c)
    return out
