from pyspark.sql import DataFrame, SparkSession


def load_csv(spark: SparkSession, p: str) -> DataFrame:
    df = (
        spark
        .read
        .format('csv')
        .options(header=True, inferSchema=True)
        .load(p)
    )
    return df


def load_parquet(spark: SparkSession, p: str) -> DataFrame:
    df = (
        spark
        .read
        .format('parquet')
        .load(p)
    )
    return df
