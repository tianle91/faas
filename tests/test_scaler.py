import pandas as pd
from pyspark.sql import DataFrame, SparkSession

from faas.scaler import NumericScaler, StandardScaler, get_mean_std


def create_test_df(spark: SparkSession, n: int = 100) -> DataFrame:
    pdf = pd.DataFrame({
        'p': ['abcdefghijklmnopqrstuvwxyz'[i % 26] for i in range(n)],
        'q': [float(i) % 100 for i in range(n)],
    })
    return spark.createDataFrame(pdf)


def test_get_mean_std(spark):
    df = create_test_df(spark)
    vals = get_mean_std(df, 'q')
    assert list(vals.keys()) == ['all'], vals
    mean, stddev = vals['all']
    assert mean > 0 and stddev > 0, str((mean, stddev))


def test_StandardScaler(spark: SparkSession):
    df = create_test_df(spark)
    expected = df.toPandas()

    sc = StandardScaler(column='q').fit(df)
    transformed_df = sc.transform(df).drop('q')
    transformed_df.collect()
    actual = sc.inverse_transform(transformed_df).select('p', 'q').toPandas()
    pd.testing.assert_frame_equal(left=expected, right=actual)

    sc = StandardScaler(column='q', group_column='p').fit(df)
    transformed_df = sc.transform(df).drop('q')
    transformed_df.collect()
    actual = sc.inverse_transform(transformed_df).select('p', 'q').toPandas()
    pd.testing.assert_frame_equal(left=expected, right=actual)


def test_NumericScaler(spark: SparkSession):
    n = 100
    df = spark.createDataFrame(pd.DataFrame({
        'q': [float(i) % 100 for i in range(n)],
        'qbase': [(float(i) % 100) + 1 for i in range(n)],
    }))
    expected = df.toPandas()
    ns = NumericScaler(column='q', base_column='qbase')
    transformed_df = ns.transform(df).drop('q')
    transformed_df.collect()
    actual = ns.inverse_transform(transformed_df).select('q', 'qbase').toPandas()
    pd.testing.assert_frame_equal(left=expected, right=actual)
