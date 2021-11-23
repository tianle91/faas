from __future__ import annotations

import logging
from typing import List, Tuple

import pandas as pd
from pyspark.sql import DataFrame
from pyspark.sql.types import DataType, DateType, NumericType, StringType

from faas.config import FeatureConfig, TargetConfig, WeightConfig
from faas.transformer.base import (AddTransformer, BaseTransformer,
                                   ConstantTransformer, Passthrough, Pipeline)
from faas.transformer.date import SeasonalityFeature
from faas.transformer.encoder import OrdinalEncoder
from faas.transformer.scaler import LogTransform, NumericScaler, StandardScaler
from faas.transformer.weight import HistoricalDecay, Normalize

logger = logging.getLogger(__name__)


def validate_types_with_msgs(
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


def validate_categorical_with_msgs(df: DataFrame, columns: List[str]) -> Tuple[bool, List[str]]:
    return validate_types_with_msgs(df=df, columns=columns, allowable_types=[StringType])


def validate_numeric_with_msgs(df: DataFrame, columns: List[str]) -> Tuple[bool, List[str]]:
    return validate_types_with_msgs(df=df, columns=columns, allowable_types=[NumericType])


def validate_date_with_msgs(df: DataFrame, columns: List[str]) -> Tuple[bool, List[str]]:
    return validate_types_with_msgs(df=df, columns=columns, allowable_types=[DateType])


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
    def __init__(self, conf: FeatureConfig):
        self.conf = conf
        self.encoded_categorical_feature_columns = []
        # create pipeline
        steps: List[BaseTransformer] = [Passthrough(columns=conf.numeric_columns)]
        for c in conf.categorical_columns:
            enc = OrdinalEncoder(c)
            self.encoded_categorical_feature_columns.append(enc.feature_column)
            steps.append(enc)
        if conf.date_column is not None:
            steps.append(SeasonalityFeature(date_column=conf.date_column))
        self.pipeline = Pipeline(steps=steps)

    def validate_input(self, df: DataFrame) -> Tuple[bool, List[str]]:
        conf = self.conf
        validations = [
            validate_numeric_with_msgs(df=df, columns=conf.numeric_columns),
            validate_categorical_with_msgs(df=df, columns=conf.categorical_columns),
        ]
        if conf.date_column is not None:
            validations.append(validate_date_with_msgs(df=df, columns=[conf.date_column]))
        return merge_validations(validations)


class YTransformer(PipelineTransformer):
    def __init__(self, conf: TargetConfig):
        self.conf = conf
        # create pipeline
        steps: List[BaseTransformer] = []
        # sequential transformations require updating current column
        c = conf.column
        if conf.log_transform:
            step = LogTransform(column=c)
            steps.append(step)
            c = step.feature_column
        if (
            conf.categorical_normalization_column is not None
            and conf.numerical_normalization_column is not None
        ):
            raise ValueError('Cannot normalize by both categorical and numerical.')
        elif conf.categorical_normalization_column is not None:
            steps.append(StandardScaler(
                column=c, group_column=conf.categorical_normalization_column))
        elif conf.numerical_normalization_column is not None:
            steps.append(NumericScaler(
                column=c, group_column=conf.numerical_normalization_column))
        else:
            # both are nones
            steps.append(Passthrough(columns=[c]))
        self.pipeline = Pipeline(steps)

    @property
    def feature_columns(self) -> List[str]:
        return [self.pipeline.feature_columns[-1]]

    def validate_input(self, df: DataFrame) -> Tuple[bool, List[str]]:
        conf = self.conf
        validations = [validate_numeric_with_msgs(df=df, columns=[conf.column])]
        if conf.categorical_normalization_column is not None:
            validations.append(validate_categorical_with_msgs(
                df=df, columns=[conf.categorical_normalization_column]))
        elif conf.numerical_normalization_column is not None:
            validations.append(validate_numeric_with_msgs(
                df=df, columns=[conf.numerical_normalization_column]))
        return merge_validations(validations)


class WTransformer(PipelineTransformer):
    def __init__(self, conf: WeightConfig):
        self.conf = conf
        # create pipeline
        steps: List[BaseTransformer] = []
        if conf.date_column is not None:
            date_steps = [
                Normalize(group_column=conf.date_column),
                HistoricalDecay(annual_rate=conf.annual_decay_rate, date_column=conf.date_column)
            ]
            steps += date_steps
        if conf.group_columns is not None:
            for c in conf.group_columns:
                steps.append(Normalize(group_column=c))
        if len(steps) > 0:
            # get the final combined weight by adding up the individual weight columns
            steps += AddTransformer(columns=[step.feature_columns[-1] for step in steps])
        else:
            steps.append(ConstantTransformer())
        self.pipeline = Pipeline(steps)

    @property
    def feature_columns(self) -> List[str]:
        return [self.pipeline.feature_columns[-1]]

    def validate_input(self, df: DataFrame) -> Tuple[bool, List[str]]:
        conf = self.conf
        validations = []
        if conf.date_column is not None:
            validations.append(validate_date_with_msgs(df=df, columns=[conf.date_column]))
        if conf.group_columns is not None:
            validations.append(validate_categorical_with_msgs(df=df, columns=conf.group_columns))
        return merge_validations(validations)
