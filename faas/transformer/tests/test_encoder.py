import pandas as pd
from pyspark.sql import DataFrame, SparkSession

from faas.transformer.encoder import OrdinalEncoder


def create_test_df(spark: SparkSession, n: int = 100) -> DataFrame:
    pdf = pd.DataFrame({
        'p': ['abcdefghijklmnopqrstuvwxyz'[i % 26] for i in range(n)],
        'q': [float(i) % 100 for i in range(n)],
    })
    return spark.createDataFrame(pdf)


def test_OrdinalEncoder(spark):
    df = create_test_df(spark)

    enc = OrdinalEncoder('p').fit(df)
    encoded_rows = enc.transform(df).distinct().collect()

    encoded_values = {row['OrdinalEncoder_p'] for row in encoded_rows}
    assert encoded_values == {i for i in range(26)}

    # check that inverse recovers original
    pdf = df.toPandas()
    pdf_identity = enc.inverse_transform(enc.transform(df).drop('p')).select(*df.columns).toPandas()
    pd.testing.assert_frame_equal(pdf, pdf_identity)
