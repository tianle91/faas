import pandas as pd
from pyspark.sql import SparkSession

from faas.eda.iid import correlation


def test_correlation(spark: SparkSession):
    n = 100
    c_l = [c for c in 'ab']
    pdf = pd.DataFrame({
        'cat': [c_l[i % len(c_l)] for i in range(n)],
        'num': [i % len(c_l) for i in range(n)],
    })
    df = spark.createDataFrame(pdf)

    cols = ['cat', 'num']
    actual = correlation(df=df, columns=cols)
    expected = pd.DataFrame([[1., 1., ], [1., 1., ]], index=cols, columns=cols)
    pd.testing.assert_frame_equal(actual, expected, rtol=1e-3)
