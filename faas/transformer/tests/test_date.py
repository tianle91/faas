from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest
from pyspark.sql import SparkSession

from faas.transformer.date import SeasonalityFeature, normalized_sine


@pytest.mark.parametrize(
    ('x', 'period', 'phase', 'expected'),
    [
        pytest.param(0, 1, 0, 0, id='sin(0)'),
        pytest.param(.25, 1, 0, 1, id='sin(pi/2)'),
    ]
)
def test_normalized_sine(x: float, period: float, phase: int, expected: float):
    actual = normalized_sine(x=x, period=period, phase=phase)
    assert np.isclose(actual, expected)


def test_DayOfWeekFeatures(spark: SparkSession):
    seasonality = SeasonalityFeature(date_column='dt')
    df = spark.createDataFrame(pd.DataFrame({
        'dt': [date(2021, 1, 1) + timedelta(days=i) for i in range(100)]
    }))
    seasonality.transform(df).toPandas()
