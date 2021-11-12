from datetime import date

import numpy as np
import pytest

from faas.date import (DayOfWeekFeatures, SeasonalityFeature, get_dom, get_dow,
                       get_doy, get_woy, normalized_sine)


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


def test_DayOfWeekFeatures():
    dowf = DayOfWeekFeatures(date_column='dt')
    assert dowf.feature_columns == [
        f'Seasonality_{DayOfWeekFeatures.period[0]}_{i}'
        for i in range(7)
    ]


def test_get_dow():
    assert get_dow(dt=date(2021, 11, 12)) == 4


def test_get_dom():
    assert get_dom(dt=date(2021, 11, 12)) == 12


def test_get_doy():
    assert get_doy(dt=date(2021, 1, 1)) == 0


def test_get_woy():
    assert get_woy(dt=date(2021, 1, 7)) == 1
