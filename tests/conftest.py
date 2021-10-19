import pandas as pd
import pytest
from pyspark.sql import DataFrame, SparkSession


@pytest.fixture(scope='session')
def spark():
    return SparkSession.builder.getOrCreate()


def create_test_df(spark: SparkSession, n: int = 1000000) -> DataFrame:
    pdf = pd.DataFrame({
        'p': ['abcdefghijklmnopqrstuvwxyz'[i % 26] for i in range(n)],
        'q': [i % 100 for i in range(n)],
    })
    return spark.createDataFrame(pdf)
