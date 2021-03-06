from datetime import datetime
from typing import List

import numpy as np
import pyspark.sql.functions as F
from pyspark.sql import DataFrame
from pyspark.sql.types import DoubleType, StringType, TimestampType

from faas.transformer.base import BaseTransformer

from .utils import validate_timestamp_types


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


class HistoricalDecay(BaseTransformer):
    """Weights with decreasing weight from 1 (newest) to 0 (infinitely old). Use if time series.
    """

    def __init__(
        self,
        annual_rate: float,
        date_column: str,
    ):
        self.annual_rate = annual_rate
        self.date_column = date_column

    @property
    def input_columns(self) -> List[str]:
        return [self.date_column]

    @property
    def feature_column(self) -> str:
        return f'HistoricalDecay_{self.date_column}'

    @property
    def feature_columns(self) -> List[str]:
        return [self.feature_column]

    def fit(self, df: DataFrame):
        validate_timestamp_types(df=df, cols=[self.date_column])
        newest_df = df.agg(F.max(F.col(self.date_column)).alias('newest'))
        self.most_recent_date = newest_df.collect()[0].newest
        return self

    def transform(self, df: DataFrame):
        validate_timestamp_types(df=df, cols=[self.date_column])
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


COUNTS_COL = '__COUNTS__'


class Normalize(BaseTransformer):
    """Weights to ensure that for each group, sum of weights is 1. Use if multivariate ts.
    """

    def __init__(self, group_column: str):
        self.group_column = group_column

    @property
    def input_columns(self) -> List[str]:
        return [self.group_column]

    @property
    def feature_column(self) -> str:
        return f'Normalize_{self.group_column}'

    @property
    def feature_columns(self) -> str:
        return [self.feature_column]

    def transform(self, df: DataFrame):

        # create dummy group col because it can be different from df group col
        DUMMY_GROUP_COL = '__DUMMY_GROUP_COL__'
        df = df.withColumn(DUMMY_GROUP_COL, F.col(self.group_column))
        dtype = df.schema[DUMMY_GROUP_COL].dataType
        # some changes due to types
        if isinstance(dtype, TimestampType):
            df = df.withColumn(
                DUMMY_GROUP_COL,
                F.to_date(F.col(DUMMY_GROUP_COL))
            )
        elif isinstance(dtype, StringType):
            pass
        else:
            raise TypeError(
                f'The group_column: {self.group_column} should be StringType or a TimestampType '
                f'but received {dtype} instead'
            )

        # create the counts
        counts = (
            df
            .groupBy(DUMMY_GROUP_COL)
            .agg(F.sum(F.lit(1.)).alias(COUNTS_COL))
        )
        df = df.join(counts, on=DUMMY_GROUP_COL, how='left')
        df = df.withColumn(self.feature_column, 1. / F.col(COUNTS_COL))

        df = df.drop(COUNTS_COL, DUMMY_GROUP_COL)
        return df
