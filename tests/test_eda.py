from faas.eda import target_correlation
import pandas as pd
from pyspark.sql import SparkSession


def test_target_correlation(spark: SparkSession):
    n = 100
    df = spark.createDataFrame(pd.DataFrame({
        'q': [float(i) % 100 for i in range(n)],
        'q1': [100 - (float(i) % 100) for i in range(n)],
    }))
    actual = target_correlation(df=df, feature_columns=['q1'], target_column='q')
    for c in df.columns:
        assert actual.loc[c, c] == 1.
        for c1 in df.columns:
            if c1 != c:
                assert actual.loc[c, c1] < 1.
