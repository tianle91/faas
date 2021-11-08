import pandas as pd
from pyspark.sql import DataFrame, SparkSession

from faas.encoder import OrdinalEncoderSingle


def create_test_df(spark: SparkSession, n: int = 100) -> DataFrame:
    pdf = pd.DataFrame({
        'p': ['abcdefghijklmnopqrstuvwxyz'[i % 26] for i in range(n)],
        'q': [float(i) % 100 for i in range(n)],
    })
    return spark.createDataFrame(pdf)


def test_OrdinalEncoderSingle(spark):
    df = create_test_df(spark)
    expected = df.toPandas()

    enc = OrdinalEncoderSingle('p').fit(df)
    encoded_rows = enc.transform(df).distinct().collect()

    encoded_values = {row.p for row in encoded_rows}
    assert encoded_values == {i for i in range(26)}

    actual = enc.inverse_transform(enc.transform(df)).toPandas()
    pd.testing.assert_frame_equal(left=expected, right=actual)
