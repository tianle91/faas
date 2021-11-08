import pandas as pd
from pyspark.sql import DataFrame, SparkSession

from faas.data import Preprocess


def create_test_df(spark: SparkSession, n: int = 100) -> DataFrame:
    pdf = pd.DataFrame({
        'q': [float(i) % 100 for i in range(n)],
        'p': ['abcdefghijklmnopqrstuvwxyz'[i % 26] for i in range(n)],
    })
    return spark.createDataFrame(pdf)


def test_Preprocess(spark):
    df = create_test_df(spark)
    expected = df.toPandas()

    pp = Preprocess(
        numeric_columns=['q'],
        categorical_columns=['p'],
    ).fit(df)
    pp.transform(df)

    actual = pp.inverse_transform(pp.transform(df)).toPandas()
    pd.testing.assert_frame_equal(left=expected, right=actual)
