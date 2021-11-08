from __future__ import annotations

from typing import Dict, Optional, Tuple

import pyspark.sql.functions as F
from pyspark.sql import DataFrame
from pyspark.sql.types import DoubleType, NumericType


def get_mean_std(
    df: DataFrame, column: str, group_column: Optional[str] = None
) -> Dict[str, Tuple[float, float]]:
    """Return {group_value: (mean, std)} of df[column] grouped by group_column (otherwise 'all').
    """
    if group_column is not None:
        if isinstance(df.schema[group_column].dataType, NumericType):
            raise TypeError(
                f'Column: {group_column} '
                f'dataType: {df.schema[group_column].dataType} is a NumericType, '
                'which cannot be used for grouping.'
            )
        df = df.groupBy(group_column)
    mean_stddevs = df.agg(
        F.mean(F.col(column)).alias('mean'),
        F.stddev(F.col(column)).alias('stddev'),
    ).withColumn(
        'stddev',
        F.when(F.col('stddev').isNull(), F.lit(1.)).otherwise(F.col('stddev'))
    ).collect()
    if group_column is not None:
        return {
            getattr(row, group_column): (getattr(row, 'mean'), getattr(row, 'stddev'))
            for row in mean_stddevs
        }
    else:
        assert len(mean_stddevs) == 1, str(mean_stddevs)
        row = mean_stddevs[0]
        return {'all': (getattr(row, 'mean'), getattr(row, 'stddev'))}


class StandardScaler:
    def __init__(self, column, group_column: Optional[str] = None) -> None:
        self.column = column
        self.group_column = group_column
        self._mean_std = None

    def fit(self, df: DataFrame) -> StandardScaler:
        self._mean_std = get_mean_std(df, column=self.column, group_column=self.group_column)
        return self

    def _get_mean_std(self, value) -> Tuple[float, float]:
        k = value if self.group_column is not None else 'all'
        mean, std = self._mean_std.get(k, (0, 1))
        return mean, std

    def transform(self, df: DataFrame) -> DataFrame:
        def fn(v) -> float:
            mean, std = self._get_mean_std(v)
            return (v - mean) / std
        return df.withColumn(self.column, F.udf(fn, DoubleType())(F.col(self.column)))

    def inverse_transform(self, df: DataFrame) -> DataFrame:
        def fn(v) -> float:
            mean, std = self._get_mean_std(v)
            return std * v + mean
        return df.withColumn(self.column, F.udf(fn, DoubleType())(F.col(self.column)))
