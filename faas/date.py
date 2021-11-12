import math
from datetime import date
from typing import Dict, List, Tuple

import pyspark.sql.functions as F
from pyspark.sql import DataFrame
from pyspark.sql.types import FloatType

from faas.base import BaseTransformer
from faas.utils_dataframe import validate_date_types


def normalized_sine(x: float, period: float, phase: int):
    x = (x + phase) / period
    return math.sin(x)


class SeasonalityFeature(BaseTransformer):

    def __init__(self, date_column: str) -> None:
        self.date_column = date_column

    def get_value(self, dt: date) -> float:
        raise NotImplementedError

    @property
    def period(self) -> Tuple[str, float]:
        raise NotImplementedError

    @property
    def feature_to_udf_mapping(self) -> Dict[str, callable]:
        out = {}
        period_name, period_value = self.period
        i = 0
        while i <= period_value:
            out[f'Seasonality_{period_name}_{i}'] = F.udf(
                lambda dt: normalized_sine(
                    self.get_value(dt),
                    period=self.period,
                    phase=i
                ),
                FloatType()
            )
            i += 1
        return out

    @ property
    def feature_columns(self) -> List[str]:
        return list(self.feature_to_udf_mapping.keys())

    def transform(self, df: DataFrame) -> DataFrame:
        validate_date_types(df, cols=[self.date_column])
        distincts = df.select(F.col(self.date_column))
        for feature_column, udf in self.feature_to_udf_mapping.items():
            distincts = distincts.withColumn(feature_column, udf(F.col(self.date_column)))
        df = df.join(distincts, on=self.date_column, how='left')
        return df


def get_dow(dt: date) -> float:
    return dt.weekday()


class DayOfWeekFeatures(BaseTransformer):
    get_value = get_dow
    period = ('WeekOfDay', 6)


def get_dom(dt: date) -> float:
    return dt.day


class DayOfMonthFeatures(BaseTransformer):
    get_value = get_dom
    period = ('DayOfMonth', 31)


def get_doy(dt: date) -> float:
    return (dt - date(dt.year, 1, 1)).days


class DayOfYearFeatures(BaseTransformer):
    get_value = get_doy
    period = ('DayOfyear', 365.25)


def get_woy(dt: date) -> float:
    return dt.isocalendar()[1]


class WeekOfYearFeatures(BaseTransformer):
    get_value = get_woy
    period = ('WeekOfYear', 52.1429)
