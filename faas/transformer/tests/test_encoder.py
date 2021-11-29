import pandas as pd
from pyspark.sql import SparkSession

from faas.transformer.encoder import OneHotEncoder, OrdinalEncoder


def test_OrdinalEncoder(spark: SparkSession):
    n = 100
    c_l = [c for c in 'abcdefghijklmnopqrstuvwxyz']
    pdf = pd.DataFrame({
        'p': [c_l[i % len(c_l)] for i in range(n)]
    })
    df = spark.createDataFrame(pdf)

    enc = OrdinalEncoder('p').fit(df)
    df_transformed = enc.transform(df).drop('p')

    # assert that encoded column is created
    assert df_transformed.columns == ['OrdinalEncoder_p']

    # assert that encoded values are as expected
    encoded_values = {row['OrdinalEncoder_p'] for row in df_transformed.distinct().collect()}
    assert encoded_values == {i for i in range(26)}

    # check that inverse recovers original
    pdf_identity = enc.inverse_transform(df_transformed).select(*df.columns).toPandas()
    pd.testing.assert_frame_equal(pdf, pdf_identity)


def test_OneHotEncoder(spark: SparkSession):
    n = 100
    c_l = ['a', 'b', 'c']
    pdf = pd.DataFrame({
        'p': [c_l[i % len(c_l)] for i in range(n)]
    })
    df = spark.createDataFrame(pdf)

    enc = OneHotEncoder('p').fit(df)
    df_transformed = enc.transform(df).drop('p')

    assert set(enc.distincts) == set(c_l)

    # assert that encoded column is created
    expected_columns = {'OneHotEncoder_p_is_a', 'OneHotEncoder_p_is_b', 'OneHotEncoder_p_is_c'}
    actual_columns = set(df_transformed.columns)
    assert actual_columns == expected_columns, actual_columns

    # assert that encoded values are as expected
    encoded_values = {
        (row['OneHotEncoder_p_is_a'], row['OneHotEncoder_p_is_b'], row['OneHotEncoder_p_is_c'])
        for row in df_transformed.distinct().collect()
    }
    expected_values = {
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
    }
    assert encoded_values == expected_values, encoded_values

    # check that inverse recovers original
    pdf_identity = enc.inverse_transform(df_transformed).select(*df.columns).toPandas()
    pd.testing.assert_frame_equal(pdf, pdf_identity)
