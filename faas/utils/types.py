
from pyspark.sql import DataFrame, SparkSession

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
