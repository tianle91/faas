from __future__ import annotations

import logging
from typing import List, Optional

import pandas as pd
from pyspark.sql import DataFrame

from faas.transformer.base import BaseTransformer, Passthrough, Pipeline
from faas.transformer.date import (DayOfMonthFeatures, DayOfWeekFeatures,
                                   DayOfYearFeatures, WeekOfYearFeatures)
from faas.transformer.encoder import OrdinalEncoder
from faas.transformer.scaler import StandardScaler, LogTransform, NumericScaler
from faas.transformer.weight import Normalize

logger = logging.getLogger(__name__)


class PipelineTransformer(BaseTransformer):

    @property
    def input_columns(self) -> List[str]:
        return self.pipeline.input_columns

    @property
    def feature_columns(self) -> List[str]:
        return self.pipeline.feature_columns

    def fit(self, df: DataFrame) -> PipelineTransformer:
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
    def __init__(
        self,
        target_column: str,
        log_transform: bool,
        normalize_by_categorical: Optional[str] = None,
        normalize_by_numerical: Optional[str] = None,
    ):
        self.target_column = target_column
        self.log_transform = log_transform
        if normalize_by_categorical is not None and normalize_by_numerical is not None:
            raise ValueError('Cannot normalize by both categorical and numerical.')
        self.normalize_by_categorical = normalize_by_categorical
        self.normalize_by_numerical = normalize_by_numerical
        # create pipeline
        steps = []
        target_column = self.target_column
        if self.log_transform:
            step = LogTransform(column=target_column)
            steps.append(step)
            target_column = step.feature_column

        if normalize_by_categorical is not None:
            steps.append(StandardScaler(
                column=target_column,
                group_column=normalize_by_categorical
            ))
        elif normalize_by_numerical is not None:
            steps.append(NumericScaler(
                column=target_column,
                group_column=normalize_by_numerical
            ))
        else:
            steps.append(Passthrough(columns=[target_column]))
        self.pipeline = Pipeline(steps)

    @property
    def feature_columns(self) -> List[str]:
        return [self.pipeline.feature_columns[-1]]


class WTransformer(PipelineTransformer):
    def __init__(self, group_columns: str):
        self.group_columns = group_columns
        steps = [Normalize(group_column=c) for c in group_columns]
        self.pipeline = Pipeline(steps)
