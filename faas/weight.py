from datetime import datetime

import numpy as np
import pyspark.sql.functions as F
from pyspark.sql import DataFrame
from pyspark.sql.types import DoubleType

from faas.utils_dataframe import (validate_categorical_types,
                                  validate_timestamp_types)


def historical_decay(annual_rate: float, today_dt: datetime, dt: datetime) -> float:
    """Discounted value of historical observations based on today_dt and dt with an annual rate.
    For dt==today_dt, the amount is 1. For dt==today_dt-1year, the amount is e^-annual_rate.
    """
    if not annual_rate >= 0.:
        raise ValueError(f'Annual decay rate: {annual_rate} must be >= 0')
    if not dt <= today_dt:
        raise ValueError(f'Historical dt: {dt} must be at least as old as today: {today_dt}')
    years_ago = (today_dt - dt).days / 360.25
    return float(np.exp(-1. * annual_rate * years_ago))


class HistoricalDecay:
    """Weights with decreasing weight from 1 (newest) to 0 (infinitely old)."""

    def __init__(
        self,
        annual_rate: float,
        timestamp_column: str,
        weight_column: str = 'HistoricalDecay'
    ):
        self.annual_rate = annual_rate
        self.timestamp_column = timestamp_column
        self.weight_column = weight_column

    def fit(self, df: DataFrame):
        validate_timestamp_types(df=df, cols=[self.timestamp_column])
        self.most_recent_date = (
            df.agg(
                F.max(F.col(self.timestamp_column)).alias('oldest')
            ).collect()[0].oldest
        )
        return self

    def transform(self, df: DataFrame):
        validate_timestamp_types(df=df, cols=[self.timestamp_column])

        def f(dt: datetime) -> float:
            return historical_decay(
                annual_rate=self.annual_rate,
                today_dt=self.most_recent_date,
                dt=dt
            )

        distincts = (
            df
            .select(self.timestamp_column)
            .distinct()
            .withColumn(
                self.weight_column,
                F.udf(f, DoubleType())(self.timestamp_column)
            )
        )
        return df.join(distincts, on=self.timestamp_column, how='left')


class Normalize:
    """Weights to ensure that for each group, sum of weights is 1."""

    def __init__(
        self,
        categorical_column: str,
        weight_column: str = 'Normalize'
    ):
        self.categorical_column = categorical_column
        self.weight_column = weight_column

    def fit(self):
        return self

    def transform(self, df: DataFrame):
        validate_categorical_types(df=df, cols=[self.categorical_column])
        COUNTS_COL = '__COUNTS__'
        counts = (
            df
            .groupBy(self.categorical_column)
            .agg(F.sum(F.lit(1.)).alias(COUNTS_COL))
        )
        df = df.join(counts, on=self.categorical_column, how='left')
        df = df.withColumn(self.weight_column, 1. / F.col(COUNTS_COL)).drop(COUNTS_COL)
        return df
