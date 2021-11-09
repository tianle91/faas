from datetime import date

import numpy as np
import pyspark.sql.functions as F
from pyspark.sql import DataFrame
from pyspark.sql.types import DoubleType

from faas.base import BaseTransformer
from faas.utils_dataframe import (validate_categorical_types,
                                  validate_date_types)


def historical_decay(annual_rate: float, today_dt: date, dt: date) -> float:
    """Discounted value of historical observations based on today_dt and dt with an annual rate.
    For dt==today_dt, the amount is 1. For dt==today_dt-1year, the amount is e^-annual_rate.
    """
    if not annual_rate >= 0.:
        raise ValueError(f'Annual decay rate: {annual_rate} must be >= 0')
    if not dt <= today_dt:
        raise ValueError(f'Historical dt: {dt} must be at least as old as today: {today_dt}')
    years_ago = (today_dt - dt).days / 360.25
    return float(np.exp(-1. * annual_rate * years_ago))


class HistoricalDecay(BaseTransformer):
    """Weights with decreasing weight from 1 (newest) to 0 (infinitely old)."""

    def __init__(
        self,
        annual_rate: float,
        date_column: str,
    ):
        self.annual_rate = annual_rate
        self.date_column = date_column

    @property
    def feature_column(self) -> str:
        return f'HistoricalDecay_{self.date_column}'

    def fit(self, df: DataFrame):
        validate_date_types(df=df, cols=[self.date_column])
        newest_df = df.agg(F.max(F.col(self.date_column)).alias('newest'))
        self.most_recent_date = newest_df.collect()[0].newest
        return self

    def transform(self, df: DataFrame):
        validate_date_types(df=df, cols=[self.date_column])
        udf = F.udf(
            lambda dt: historical_decay(
                annual_rate=self.annual_rate,
                today_dt=self.most_recent_date,
                dt=dt
            ),
            DoubleType()
        )
        distincts = df.select(self.date_column).distinct()
        distincts = distincts.withColumn(self.feature_column, udf(self.date_column))
        return df.join(distincts, on=self.date_column, how='left')


class Normalize(BaseTransformer):
    """Weights to ensure that for each group, sum of weights is 1."""

    def __init__(
        self,
        categorical_column: str,
    ):
        self.categorical_column = categorical_column

    @property
    def feature_column(self) -> str:
        return f'Normalize_{self.categorical_column}'

    def transform(self, df: DataFrame):
        validate_categorical_types(df=df, cols=[self.categorical_column])
        COUNTS_COL = '__COUNTS__'
        counts = (
            df
            .groupBy(self.categorical_column)
            .agg(F.sum(F.lit(1.)).alias(COUNTS_COL))
        )
        df = df.join(counts, on=self.categorical_column, how='left')
        df = df.withColumn(self.feature_column, 1. / F.col(COUNTS_COL)).drop(COUNTS_COL)
        return df
