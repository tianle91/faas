import pandas as pd
from pyspark.sql import SparkSession

from faas.transformer.base import AddTransformer


def test_AddTransformer(spark: SparkSession):
    pdf = pd.DataFrame({
        'a': [1, 2, 3],
        'b': [4, 5, 6],
    })
    at = AddTransformer(columns=['a', 'b'])
    actual = at.transform(spark.createDataFrame(pdf)).toPandas()
    expected = pdf.copy()
    expected['AddTransformer_a+b'] = [5, 7, 9]
    pd.testing.assert_frame_equal(actual, expected)
