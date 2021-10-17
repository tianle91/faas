import pandas as pd
import pytest
from pyspark.sql import SparkSession


@pytest.mark.parametrize(
    ('arg1', 'arg2'),
    [
        pytest.param(1, 2, id='case 1'),
        pytest.param(2, 3, id='case 2'),
    ]
)
def test_function(arg1: int, arg2: int):
    assert isinstance(arg1, int) and isinstance(arg2, int)


def test_spark(spark: SparkSession):
    df = pd.DataFrame({'a': range(100), 'b': ['A', 'B'] * 50})
    sdf = spark.createDataFrame(df)
    actual = sdf.groupBy('b').count().orderBy('b').toPandas()
    expected = pd.DataFrame({'b': ['A', 'B'], 'count': [50, 50]})
    pd.testing.assert_frame_equal(expected, actual)
