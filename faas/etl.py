from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import pandas as pd
from pyspark.sql import DataFrame
from pyspark.sql.types import DataType, DateType, NumericType, StringType

from faas.transformer.base import (AddTransformer, BaseTransformer,
                                   Passthrough, Pipeline)
from faas.transformer.date import (DayOfMonthFeatures, DayOfWeekFeatures,
                                   DayOfYearFeatures, WeekOfYearFeatures)
from faas.transformer.encoder import OrdinalEncoder
from faas.transformer.scaler import LogTransform, NumericScaler, StandardScaler
from faas.transformer.weight import Normalize

logger = logging.getLogger(__name__)


def validate_types(
    df: DataFrame, columns: List[str], allowable_types: List[DataType],
) -> Tuple[bool, List[str]]:
    ok, msgs = True, []
    for c in columns:
        actual_dtype = df.schema[c].dataType
        ok_temp = any([isinstance(actual_dtype, dtype) for dtype in allowable_types])
        if not ok_temp:
            msgs.append(
                f'Expected one of {allowable_types} for column: {c} '
                f'but received {actual_dtype} instead.'
            )
        ok = ok and ok_temp
    return ok, msgs


def merge_validations(validations: List[Tuple[bool, List[str]]]) -> Tuple[bool, List[str]]:
    ok, msgs = True, []
    for ok_temp, msgs_temp in validations:
        ok = ok and ok_temp
        msgs += msgs_temp
    return ok, msgs


class PipelineTransformer(BaseTransformer):

    @property
    def input_columns(self) -> List[str]:
        return self.pipeline.input_columns

    @property
    def feature_columns(self) -> List[str]:
        return self.pipeline.feature_columns

    def validate_input(self, df: DataFrame) -> Tuple[bool, List[str]]:
        raise NotImplementedError

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

    @property
    def categorical_feature_columns(self) -> List[str]:
        out = []
        for step in self.pipeline.steps:
            if isinstance(step, OrdinalEncoder):
                out.append(step.feature_column)
        return out

    def validate_input(self, df: DataFrame) -> Tuple[bool, List[str]]:
        validations = [
            validate_types(df=df, columns=self.numeric_features, allowable_types=[NumericType]),
            validate_types(df=df, columns=self.categorical_features, allowable_types=[StringType]),
        ]
        if self.date_column is not None:
            validations.append(
                validate_types(df=df, columns=self.date_column, allowable_types=[DateType]))
        return merge_validations(validations)


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

    def validate_input(self, df: DataFrame) -> Tuple[bool, List[str]]:
        validations = [
            validate_types(df=df, columns=[self.target_column], allowable_types=[NumericType]),
        ]
        if self.normalize_by_categorical is not None:
            validations.append(validate_types(
                df=df, columns=self.normalize_by_categorical, allowable_types=[StringType]))
        elif self.normalize_by_numerical is not None:
            validations.append(validate_types(
                df=df, columns=self.normalize_by_numerical, allowable_types=[NumericType]))
        return merge_validations(validations)


class WTransformer(PipelineTransformer):
    def __init__(self, group_columns: str):
        self.group_columns = group_columns
        steps = [Normalize(group_column=c) for c in group_columns]
        if len(steps) > 1:
            wgt_cols = []
            for step in steps:
                wgt_cols += step.feature_columns
            # adding preserves group summation equality
            steps += AddTransformer(columns=wgt_cols)
        self.pipeline = Pipeline(steps)

    @property
    def feature_columns(self) -> List[str]:
        return [self.pipeline.feature_columns[-1]]

    def validate_input(self, df: DataFrame) -> Tuple[bool, List[str]]:
        return validate_types(df=df, columns=self.group_columns, allowable_types=[NumericType])
