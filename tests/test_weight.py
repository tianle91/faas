from datetime import datetime

import numpy as np
import pytest

from faas.weight import historical_decay


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
