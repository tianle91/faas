import pandas as pd
from pyspark.sql import SparkSession

from faas.stats import correlation


def test_correlation(spark: SparkSession):
    n = 100
    c_l = ['a', 'b', 'c']
    v_l = [0, 0, 1]
    pdf = pd.DataFrame({
        'cat': [c_l[i % len(c_l)] for i in range(n)],
        'num': [v_l[i % len(v_l)] for i in range(n)],
    })
    df = spark.createDataFrame(pdf)

    cols = ['cat', 'num']
    actual = correlation(df=df, columns=cols, collapse_categorical=True)
    expected = pd.DataFrame(
        [
            [1., 1., ],
            [1., 1., ]
        ],
        index=cols,
        columns=cols
    )
    pd.testing.assert_frame_equal(actual, expected, rtol=1e-3)
