from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest
from pyspark.sql import SparkSession

from faas.transformer.weight import HistoricalDecay, Normalize, historical_decay


@pytest.mark.parametrize(
    ('today_dt', 'dt', 'expected'),
    [
        pytest.param(
            date(2000, 1, 1), date(2000, 1, 1), 1., id='0 years'
        ),
        pytest.param(
            date(2000, 1, 1), date(1999, 1, 1), float(np.exp(-1)), id='1 year'
        ),
    ]
)
def test_historical_decay(today_dt: date, dt: date, expected: float):
    actual = historical_decay(annual_rate=1., today_dt=today_dt, dt=dt)
    # loose tolerance due to inexactness of a single year, assumed to be 360.25 days
    assert np.isclose(actual, expected, atol=1e-2)


def test_HistoricalDecay(spark: SparkSession):
    df = spark.createDataFrame(pd.DataFrame({
        'dt': [date(2000, 1, 1) - timedelta(weeks=i) for i in range(100)],
    }))
    hd = HistoricalDecay(annual_rate=100., date_column='dt').fit(df)
    actual = hd.transform(df).toPandas()
    assert max(actual[hd.feature_column]) == 1.
    assert actual.loc[actual['dt'] == date(2000, 1, 1), hd.feature_column].iloc[0] == 1.
    assert np.isclose(min(actual[hd.feature_column]), 0.)


def test_Normalize(spark: SparkSession):
    df = spark.createDataFrame(pd.DataFrame({
        'col': ['A', 'A', 'A', 'A', 'B', 'B', 'C'],
    }))
    nm = Normalize(group_column='col')
    actual = nm.transform(df).orderBy('col').toPandas()
    expected = pd.DataFrame({
        'col': ['A', 'A', 'A', 'A', 'B', 'B', 'C'],
        nm.feature_column: [.25, .25, .25, .25, .5, .5, 1.],
    })
    pd.testing.assert_frame_equal(expected, actual)
