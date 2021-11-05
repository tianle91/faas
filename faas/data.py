from __future__ import annotations

import logging
from typing import List, Optional

import pyspark.sql.functions as F
from lightgbm import LGBMRegressor
from pyspark.sql import DataFrame
from pyspark.sql.types import DoubleType, NumericType

from faas.encoder import OrdinalEncoderSingleSpark
from faas.scaler import StandardScalerSpark

logger = logging.getLogger(__name__)

ROW_ID_COL = '__ROW_ID__'


def get_non_numeric_columns(df: DataFrame) -> List[str]:
    return [c for c in df.columns if not isinstance(df.schema[c].dataType, NumericType)]


class Preprocess:
    """Preprocess data.

    1. Convert categorical data into ordinal encoding using OrdinalEncoder.
    2. (TBD) Scale target features within each group using StandardScaler.
    3. (TBD) add date-related features.
    4. (TBD) add location-related features.
    """

    def __init__(
        self,
        categorical_columns: Optional[List[str]] = None,
        target_scaling_group_by_column: Optional[str] = None,
        target_column: Optional[str] = None
    ):
        self.encoder = {c: OrdinalEncoderSingleSpark(c) for c in categorical_columns}
        if target_column is None and target_scaling_group_by_column is not None:
            raise ValueError(
                f'Cannot specify '
                f'target_scaling_group_by_column: {target_scaling_group_by_column} '
                f'without specifying target_column.'
            )
        self.scaler = None
        if target_column is not None:
            self.scaler = StandardScalerSpark(
                column=target_column, group_column=target_scaling_group_by_column)

    def fit(self, df: DataFrame) -> Preprocess:
        if self.scaler is not None:
            self.scaler.fit(df)
        for enc in self.encoder.values():
            enc.fit(df)
        return self

    def transform_X(self, df: DataFrame) -> DataFrame:
        for enc in self.encoder.values():
            df = enc.transform(df)
        return df

    def transform(self, df: DataFrame) -> DataFrame:
        if self.scaler is not None:
            df = self.scaler.transform(df)
        df = self.transform_X(df)
        return df

    def inverse_transform_X(self, df: DataFrame) -> DataFrame:
        for enc in self.encoder.values():
            df = enc.inverse_transform(df)
        return df

    def inverse_transform(self, df: DataFrame) -> DataFrame:
        df = self.inverse_transform_X(df)
        if self.scaler is not None:
            df = self.scaler.inverse_transform(df)
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
            target_scaling_group_by_column=target_scaling_group_by_column,
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
