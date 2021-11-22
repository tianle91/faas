import math
from datetime import date
from typing import Dict, List, Tuple

import pyspark.sql.functions as F
from pyspark.sql import DataFrame
from pyspark.sql.types import FloatType

from faas.transformer.base import BaseTransformer

from .utils import validate_date_types


def normalized_sine(x: float, period: float, phase: int):
    x = 2. * math.pi * (x + phase) / period
    return math.sin(x)


def normalized_cosine(x: float, period: float, phase: int):
    x = 2. * math.pi * (x + phase) / period
    return math.cos(x)


class SeasonalityFeature(BaseTransformer):
    """Use when there are date columns.
    """

    def __init__(self, date_column: str) -> None:
        self.date_column = date_column

    @property
    def input_columns(self) -> List[str]:
        return [self.date_column]

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
        while i <= (period_value / 2.):
            out[f'Seasonality_{period_name}_sin_{i}'] = F.udf(
                lambda dt: normalized_sine(
                    self.get_value(dt),
                    period=period_value,
                    phase=i
                ),
                FloatType()
            )
            out[f'Seasonality_{period_name}_cos_{i}'] = F.udf(
                lambda dt: normalized_cosine(
                    self.get_value(dt),
                    period=period_value,
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


class DayOfWeekFeatures(SeasonalityFeature):
    period = ('DayOfWeek', 6)

    def get_value(self, dt: date) -> float:
        return dt.weekday()


class DayOfMonthFeatures(SeasonalityFeature):
    period = ('DayOfMonth', 31)

    def get_value(self, dt: date) -> float:
        return dt.day


class DayOfYearFeatures(SeasonalityFeature):
    period = ('DayOfyear', 365.25)

    def get_value(self, dt: date) -> float:
        return (dt - date(dt.year, 1, 1)).days


class WeekOfYearFeatures(SeasonalityFeature):
    period = ('WeekOfYear', 52.1429)

    def get_value(self, dt: date) -> float:
        return dt.isocalendar()[1]
