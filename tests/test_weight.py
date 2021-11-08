from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from pyspark.sql import DataFrame, SparkSession

from faas.weight import HistoricalDecay, historical_decay


@pytest.mark.parametrize(
    ('today_dt', 'dt', 'expected'),
    [
        pytest.param(
            datetime(2000, 1, 1), datetime(2000, 1, 1), 1., id='0 years'
        ),
        pytest.param(
            datetime(2000, 1, 1), datetime(1999, 1, 1), float(np.exp(-1)), id='1 year'
        ),
    ]
)
def test_historical_decay(today_dt: datetime, dt: datetime, expected: float):
    actual = historical_decay(annual_rate=1., today_dt=today_dt, dt=dt)
    # loose tolerance due to inexactness of a single year, assumed to be 360.25 days
    assert np.isclose(actual, expected, atol=1e-2)


def create_test_df(spark: SparkSession, n: int = 100) -> DataFrame:
    pdf = pd.DataFrame({
        'dt': [datetime(2000, 1, 1) - timedelta(weeks=i) for i in range(n)],
    })
    return spark.createDataFrame(pdf)


def test_HistoricalDecay(spark: SparkSession):
    df = create_test_df(spark)
    hd = HistoricalDecay(annual_rate=100., timestamp_column='dt', weight_column='hd')
    actual = hd.fit(df).transform(df).toPandas()
    assert max(actual['hd']) == 1.
    assert actual.loc[actual['dt'] == datetime(2000, 1, 1), 'hd'].iloc[0] == 1.
    assert np.isclose(min(actual['hd']), -1.)
