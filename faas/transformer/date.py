import math
from datetime import date
from typing import Dict, List, Tuple, Union

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


SEASONALITY_FEATURE_MAPPING = {
    'day_of_week': (lambda dt: dt.weekday(), 6, 1),
    'day_of_month': (lambda dt: dt.day, 31, 3),
    'week_of_year': (lambda dt: dt.isocalendar()[1], 52.1429, 7),
    'day_of_year': (lambda dt: (dt - date(dt.year, 1, 1)).days, 365.25, 13),
}


class SeasonalityFeature(BaseTransformer):
    """Use when there are date columns.
    """

    def __init__(
        self,
        date_column: str,
        seasonality_features: Union[str, List[str]] = 'all',
    ) -> None:
        self.date_column = date_column
        self.seasonality_features = seasonality_features
        if seasonality_features == 'all':
            self.seasonality_features = list(SEASONALITY_FEATURE_MAPPING.keys())

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
        for k in self.seasonality_features:
            period_fn, period_value, period_skip = SEASONALITY_FEATURE_MAPPING[k]
            i = 0
            while i <= (period_value / 2.):
                out[f'Seasonality_{k}_sin_{i}'] = F.udf(
                    lambda dt: normalized_sine(period_fn(dt), period=period_value, phase=i),
                    FloatType()
                )
                out[f'Seasonality_{k}_cos_{i}'] = F.udf(
                    lambda dt: normalized_cosine(period_fn(dt), period=period_value, phase=i),
                    FloatType()
                )
                i += period_skip
        return out

    @ property
    def feature_columns(self) -> List[str]:
        return list(self.feature_to_udf_mapping.keys())

    def transform(self, df: DataFrame) -> DataFrame:
        validate_date_types(df, cols=[self.date_column])
        distincts = df.select(F.col(self.date_column)).distinct()
        for feature_column, udf in self.feature_to_udf_mapping.items():
            distincts = distincts.withColumn(feature_column, udf(F.col(self.date_column)))
        df = df.join(distincts, on=self.date_column, how='left')
        return df
