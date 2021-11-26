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
