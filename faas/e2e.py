from __future__ import annotations

import logging
from typing import List, Optional, Tuple

from lightgbm import LGBMModel
from pyspark.sql import DataFrame
from pyspark.sql.types import NumericType, StringType

from faas.etl import WTransformer, XTransformer, YTransformer
from faas.utils.dataframe import (JoinableByRowID,
                                  check_columns_are_desired_type)

logger = logging.getLogger(__name__)


class E2EPipline:

    def __init__(
        self,
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

        self.xtransformer = XTransformer(
            numeric_features=self.numeric_features,
            categorical_features=self.categorical_features,
            date_column=self.date_column
        )
        self.ytransformer = YTransformer(
            target_column=self.target_column,
            group_column=self.target_group_column
        )
        self.wtransformer = None
        if self.date_column is not None:
            self.wtransformer = WTransformer(date_column=self.date_column)

        self.m = LGBMModel(objective='regression', deterministic=True)

    @property
    def feature_columns(self) -> List[str]:
        out = []
        if self.categorical_features is not None:
            out += self.categorical_features
        if self.numeric_features is not None:
            out += self.numeric_features
        return out

    def check_target(self, df: DataFrame) -> Tuple[bool, List[str]]:
        return check_columns_are_desired_type(
            columns=[self.target_column], dtype=NumericType, df=df)

    def check_numeric(self, df: DataFrame) -> Tuple[bool, List[str]]:
        return check_columns_are_desired_type(
            columns=self.numeric_features, dtype=NumericType, df=df)

    def check_categorical(self, df: DataFrame) -> Tuple[bool, List[str]]:
        return check_columns_are_desired_type(
            columns=self.categorical_features, dtype=StringType, df=df)

    def fit(self, df: DataFrame) -> E2EPipline:
        X = self.xtransformer.fit(df).get_transformed_as_pdf(df)
        y = self.ytransformer.fit(df).get_transformed_as_pdf(df)
        p = {}
        if self.wtransformer is not None:
            p['sample_weight'] = self.wtransformer.fit(df).get_transformed_as_pdf(df)

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
        Xpred = self.xtransformer.get_transformed_as_pdf(jb.df)
        ypred = self.m.predict(Xpred)
        df_with_y = jb.join_by_row_id(
            ypred,
            column=self.ytransformer.feature_columns[0]
        )
        df_pred = self.ytransformer.inverse_transform(df_with_y)
        return df_pred
