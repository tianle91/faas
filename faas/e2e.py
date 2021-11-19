from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from lightgbm import LGBMModel
from matplotlib.figure import Figure
from pyspark.sql import DataFrame
from pyspark.sql.types import NumericType, StringType

from faas.transformer.base import Passthrough, Pipeline
from faas.transformer.date import (DayOfMonthFeatures, DayOfWeekFeatures,
                                   DayOfYearFeatures, WeekOfYearFeatures)
from faas.transformer.encoder import OrdinalEncoder
from faas.transformer.scaler import StandardScaler
from faas.transformer.weight import Normalize
from faas.utils.dataframe import (JoinableByRowID,
                                  check_columns_are_desired_type)

logger = logging.getLogger(__name__)


def get_x_pipeline(
    numeric_features: List[str],
    categorical_features: List[str],
    date_column: Optional[str] = None
) -> Pipeline:
    xsteps = [Passthrough(columns=numeric_features)]
    xsteps += [OrdinalEncoder(c) for c in categorical_features]
    if date_column is not None:
        xsteps += [
            DayOfMonthFeatures(date_column=date_column),
            DayOfWeekFeatures(date_column=date_column),
            DayOfYearFeatures(date_column=date_column),
            WeekOfYearFeatures(date_column=date_column),
        ]
    return Pipeline(steps=xsteps)


def get_y_pipeline(
    target_column: str, target_is_numeric: bool = True, group_column: Optional[str] = None,
) -> Pipeline:
    if not target_is_numeric:
        return Pipeline([OrdinalEncoder(categorical_column=target_column)])
    if group_column is not None:
        return Pipeline([StandardScaler(column=target_column, group_column=group_column)])
    else:
        return Pipeline([Passthrough(columns=[target_column])])


def get_w_pipeline(date_column: str) -> Pipeline:
    return Pipeline([Normalize(group_column=date_column)])


class E2EPipline:

    def __init__(
        self,
        # x_pipeline: Pipeline,
        # y_pipeline: Pipeline,
        # w_pipeline: Pipeline,
        df: DataFrame,
        target_column: str,
        target_group_column: Optional[str] = None,
        date_column: Optional[str] = None,
        categorical_features: Optional[List[str]] = None,
        numeric_features: Optional[List[str]] = None,
    ):
        self.target_column = target_column
        if not isinstance(df.schema[self.target_column].dataType, NumericType):
            raise TypeError(f'Currently only supporting numeric target: {target_column}')
        self.target_group_column = target_group_column

        self.date_column = date_column
        self.categorical_features = categorical_features
        self.numeric_features = numeric_features
        logger.info(
            f'num_categorical_features: {len(categorical_features)} '
            f'num_numeric_features: {len(numeric_features)}'
        )

        self.x_pipeline = get_x_pipeline(
            numeric_features=self.numeric_features,
            categorical_features=self.categorical_features,
            date_column=self.date_column
        )
        self.y_pipeline = get_y_pipeline(
            target_column=self.target_column,
            target_is_numeric=self.target_is_numeric,
            group_column=self.target_group_column
        )
        self.w_pipeline = None
        if self.date_column is not None:
            self.w_pipeline = get_w_pipeline(date_column=self.date_column)

        self.m = LGBMModel(
            objective='regression' if self.target_is_numeric else 'binary',
            deterministic=True,
        )

    @property
    def feature_columns(self) -> List[str]:
        out = []
        if self.categorical_features is not None:
            out += self.categorical_features
        if self.numeric_features is not None:
            out += self.numeric_features
        return out

    def check_target(self, df: DataFrame) -> Tuple[bool, List[str]]:
        expected_dtype = NumericType if self.target_is_numeric else StringType
        return check_columns_are_desired_type(
            columns=[self.target_column], dtype=expected_dtype, df=df)

    def check_numeric(self, df: DataFrame) -> Tuple[bool, List[str]]:
        return check_columns_are_desired_type(
            columns=self.numeric_features, dtype=NumericType, df=df)

    def check_categorical(self, df: DataFrame) -> Tuple[bool, List[str]]:
        return check_columns_are_desired_type(
            columns=self.categorical_features, dtype=StringType, df=df)

    def fit(self, df: DataFrame) -> E2EPipline:
        X = self.x_pipeline.fit(df).get_transformed_as_pdf(df)
        y = self.y_pipeline.fit(df).get_transformed_as_pdf(df)
        p = {}
        if self.w_pipeline is not None:
            p['sample_weight'] = self.w_pipeline.fit(df).get_transformed_as_pdf(df)

        self.m.fit(
            X=X,
            y=y,
            feature_name=self.feature_columns,
            categorical_feature=self.categorical_features,
            **p
        )
        return self

    def predict(self, df: DataFrame) -> DataFrame:
        jb = JoinableByRowID(df)
        Xpred = self.x_pipeline.get_transformed_as_pdf(jb.df)
        ypred = self.m.predict(Xpred)
        df_with_y = jb.join_by_row_id(
            ypred,
            column=self.y_pipeline.feature_columns[0]
        )
        df_pred = self.y_pipeline.inverse_transform(df_with_y)
        return df_pred


def plot_feature_importances(m: LGBMModel, top_n: int = 10) -> Figure:
    df = pd.DataFrame({
        'name': m.feature_name_,
        'importance': m.feature_importances_,
    }).sort_values('importance', ascending=False)
    top_df = df.iloc[:top_n]
    with plt.xkcd():
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(x=top_df['name'], height=top_df['importance'])
        ax.tick_params(labelrotation=90)
        ax.set_title('Feature Importances')
    return fig
