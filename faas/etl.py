from __future__ import annotations

import logging
from typing import List, Optional

import pandas as pd
from pyspark.sql import DataFrame

from faas.transformer.base import BaseTransformer, Passthrough, Pipeline
from faas.transformer.date import (DayOfMonthFeatures, DayOfWeekFeatures,
                                   DayOfYearFeatures, WeekOfYearFeatures)
from faas.transformer.encoder import OrdinalEncoder
from faas.transformer.scaler import StandardScaler
from faas.transformer.weight import Normalize

logger = logging.getLogger(__name__)


class PipelineTransformer(BaseTransformer):

    @property
    def input_columns(self) -> List[str]:
        return self.pipeline.input_columns

    @property
    def feature_columns(self) -> List[str]:
        return self.pipeline.feature_columns

    def fit(self, df: DataFrame) -> XTransformer:
        self.pipeline.fit(df)
        return self

    def transform(self, df: DataFrame) -> DataFrame:
        return self.pipeline.transform(df)

    def get_transformed_as_pdf(self, df: DataFrame) -> pd.DataFrame:
        return self.transform(df).select(*self.feature_columns).toPandas()

    def inverse_transform(self, df: DataFrame) -> DataFrame:
        return self.pipeline.inverse_transform(df)


class XTransformer(PipelineTransformer):
    def __init__(
        self,
        numeric_features: List[str],
        categorical_features: List[str],
        date_column: Optional[str] = None
    ):
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.date_column = date_column
        # create pipeline
        xsteps = [Passthrough(columns=numeric_features)]
        xsteps += [OrdinalEncoder(c) for c in categorical_features]
        if self.date_column is not None:
            xsteps += [
                DayOfMonthFeatures(date_column=date_column),
                DayOfWeekFeatures(date_column=date_column),
                DayOfYearFeatures(date_column=date_column),
                WeekOfYearFeatures(date_column=date_column),
            ]
        self.pipeline = Pipeline(steps=xsteps)


class YTransformer(PipelineTransformer):
    def __init__(self, target_column: str, group_column: Optional[str] = None):
        self.target_column = target_column
        self.group_column = group_column
        # create pipeline
        steps = []
        if group_column is not None:
            steps.append(StandardScaler(column=target_column, group_column=group_column))
        else:
            steps.append(Passthrough(columns=[target_column]))
        self.pipeline = Pipeline(steps)


class WTransformer(PipelineTransformer):
    def __init__(self, date_column: str):
        self.date_column = date_column
        self.pipeline = Pipeline([Normalize(group_column=date_column)])
