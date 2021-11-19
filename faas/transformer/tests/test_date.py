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
        'Seasonality_DayOfWeek_sin_0',
        'Seasonality_DayOfWeek_cos_0',
        'Seasonality_DayOfWeek_sin_1',
        'Seasonality_DayOfWeek_cos_1',
        'Seasonality_DayOfWeek_sin_2',
        'Seasonality_DayOfWeek_cos_2',
        'Seasonality_DayOfWeek_sin_3',
        'Seasonality_DayOfWeek_cos_3',
    ]
    assert dowf.feature_columns == expected_feature_columns
    df = spark.createDataFrame(pd.DataFrame({
        'dt': [date(2021, 1, 1) + timedelta(days=i) for i in range(100)]
    }))
    df = dowf.transform(df)
    assert set(df.columns) == (set(expected_feature_columns) | {'dt', })
    df.collect()
