from __future__ import annotations

import logging
from typing import List, Optional

import pyspark.sql.functions as F
from lightgbm import LGBMRegressor
from pyspark.sql import DataFrame
from pyspark.sql.types import DoubleType, NumericType, StringType

from faas.encoder import OrdinalEncoderSingleSpark
from faas.scaler import StandardScalerSpark

logger = logging.getLogger(__name__)

ROW_ID_COL = '__ROW_ID__'


def get_non_numeric_columns(df: DataFrame) -> List[str]:
    return [c for c in df.columns if not isinstance(df.schema[c].dataType, NumericType)]


def validate_numeric_types(df: DataFrame, cols: List[str]):
    for c in cols:
        dtype = df.schema[c].dataType
        if not isinstance(dtype, NumericType):
            raise TypeError(f'Column {c} is {dtype} but is expected to be numeric.')


def validate_categorical_types(df: DataFrame, cols: List[str]):
    for c in cols:
        dtype = df.schema[c].dataType
        if not isinstance(dtype, StringType):
            raise TypeError(f'Column {c} is {dtype} but is expected to be string.')


class GetX:
    """Get covariates."""

    def __init__(
        self,
        numeric_columns: Optional[List[str]] = None,
        categorical_columns: Optional[List[str]] = None,
    ):
        self.numeric_columns = []
        if numeric_columns is not None:
            self.numeric_columns = numeric_columns.copy()
        self.encoder = {}
        if categorical_columns is not None:
            self.encoder = {
                c: OrdinalEncoderSingleSpark(c)
                for c in categorical_columns
            }

    def _validate_transform(self, df: DataFrame):
        validate_numeric_types(df, cols=self.numeric_columns)
        validate_categorical_types(df, cols=self.encoder.keys())

    def fit(self, df: DataFrame) -> Preprocess:
        self._validate_transform(df)
        for enc in self.encoder.values():
            enc.fit(df)
        return self

    def transform(self, df: DataFrame) -> DataFrame:
        self._validate_transform(df)
        df = df.select(*self.numeric_columns, *self.encoder.keys())
        for enc in self.encoder.values():
            df = enc.transform(df)
        return df

    def _validate_inverse_transform(self, df: DataFrame):
        validate_numeric_types(df, cols=self.numeric_columns)
        validate_numeric_types(df, cols=self.encoder.keys())

    def inverse_transform(self, df: DataFrame) -> DataFrame:
        self._validate_inverse_transform(df)
        df = df.select(*self.numeric_columns, *self.encoder.keys())
        for enc in self.encoder.values():
            df = enc.inverse_transform(df)
        return df


class GetY:
    """Get target."""

    def __init__(
        self,
        scaling_by_column: Optional[str] = None,
        target_column: Optional[str] = None
    ):
        if target_column is None and scaling_by_column is not None:
            raise ValueError(
                f'Cannot specify '
                f'scaling_by_column: {scaling_by_column} '
                f'without specifying target_column.'
            )
        self.scaler = None
        if target_column is not None:
            self.scaler = StandardScalerSpark(
                column=target_column, group_column=scaling_by_column)

    def fit(self, df: DataFrame) -> Preprocess:
        if self.scaler is not None:
            self.scaler.fit(df)
        return self

    def transform(self, df: DataFrame) -> DataFrame:
        if self.scaler is not None:
            df = self.scaler.transform(df)
        return df

    def inverse_transform(self, df: DataFrame) -> DataFrame:
        if self.scaler is not None:
            df = self.scaler.inverse_transform(df)
        return df


class Preprocess:
    """Preprocess data."""

    def __init__(
        self,
        numeric_columns: Optional[List[str]] = None,
        categorical_columns: Optional[List[str]] = None,
        scale_by_column: Optional[str] = None,
        target_column: Optional[str] = None
    ):
        self.GetX = GetX(numeric_columns=numeric_columns, categorical_columns=categorical_columns)
        self.GetY = GetY(scaling_by_column=scale_by_column, target_column=target_column)

    def fit(self, df: DataFrame) -> Preprocess:
        self.GetX.fit(df)
        self.GetY.fit(df)
        return self

    def transform(self, df: DataFrame) -> DataFrame:
        df = self.GetY.transform(df)
        df = self.GetX.transform(df)
        return df

    def inverse_transform(self, df: DataFrame) -> DataFrame:
        df = self.GetX.inverse_transform(df)
        df = self.GetY.inverse_transform(df)
        return df


class LGBMDataWrapper:
    def __init__(
        self,
        target_column: str,
        covariate_columns: List[str],
        categorical_columns: Optional[List[str]],
        target_scaling_group_by_column: Optional[str],
        **lgbm_params
    ) -> None:
        self.target_column = target_column
        self.covariate_columns = covariate_columns
        self.pp = Preprocess(
            categorical_columns=categorical_columns,
            target_column=target_column,
            scale_by_column=target_scaling_group_by_column,
        )
        self.m = LGBMRegressor(**lgbm_params)

    def fit(self, df: DataFrame) -> LGBMDataWrapper:
        df = df.select(*self.covariate_columns, self.target_column)
        # update Preprocess and apply transforms
        self.pp.fit(df)
        xy = self.pp.transform(df).toPandas()
        # train model with X, y
        X, y = xy[self.covariate_columns], xy[self.target_column]
        self.m.fit(X, y)
        return self

    def predict(self, df: DataFrame):
        df = df.select(*self.covariate_columns)
        # insert an index because we need to join predicted results back
        df = df.withColumn(ROW_ID_COL, F.monotonically_increasing_id())
        pdf = self.pp.transform_X(df).toPandas()
        ypreds = self.m.predict(pdf[self.covariate_columns])
        # add target column into df
        ypred_mapping = {i: float(ypred) for i, ypred in zip(pdf[ROW_ID_COL], ypreds)}
        df = df.withColumn(
            self.target_column,
            F.udf(lambda i: ypred_mapping[i], DoubleType())(F.col(ROW_ID_COL))
        )
        # inverse the transformations
        df = self.pp.inverse_transform_X(df)
        df = self.pp.inverse_transform_y(df)
        return df
