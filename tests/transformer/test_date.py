from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest
from pyspark.sql import SparkSession

from faas.transformer.date import DayOfWeekFeatures, normalized_sine


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
    dowf = DayOfWeekFeatures(date_column='dt')
    expected_feature_columns = [
        f'Seasonality_{DayOfWeekFeatures.period[0]}_{i}'
        for i in range(7)
    ]
    assert dowf.feature_columns == expected_feature_columns
    df = spark.createDataFrame(pd.DataFrame({
        'dt': [date(2021, 1, 1) + timedelta(days=i) for i in range(100)]
    }))
    df = dowf.transform(df)
    assert set(df.columns) == (set(expected_feature_columns) | {'dt', })
    df.collect()
