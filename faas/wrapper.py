from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import pyspark.sql.functions as F
from lightgbm import LGBMModel
from pyspark.sql import DataFrame
from pyspark.sql.types import DateType, NumericType, StringType

from faas.eda.iid import correlation
from faas.etl import (WTransformer, XTransformer, YTransformer,
                      merge_validations)
from faas.utils.dataframe import JoinableByRowID


@dataclass
class ETLConfig:
    x_categorical_columns: List[str]
    x_numeric_features: List[str]
    target_column: str
    target_log_transform: bool = False
    target_normalize_by_categorical: Optional[str] = None
    target_normalize_by_numerical: Optional[str] = None
    date_column: Optional[str] = None
    weight_group_columns: Optional[List[str]] = None


class ETLWrapperForLGBM:

    def __init__(self, config: ETLConfig):
        self.ytransformer = YTransformer(
            target_column=config.target_column,
            log_transform=config.target_log_transform,
            normalize_by_categorical=config.target_normalize_by_categorical,
            normalize_by_numerical=config.target_normalize_by_numerical,
        )
        self.xtransformer = XTransformer(
            numeric_features=config.x_numeric_features,
            categorical_features=config.x_categorical_columns,
            date_column=config.date_column
        )
        self.wtransformer = None
        if config.weight_group_columns is not None:
            self.wtransformer = WTransformer(group_columns=config.weight_group_columns)
        self.m = LGBMModel(objective='regression', deterministic=True)

    def check_df_prediction(self, df: DataFrame) -> Tuple[bool, List[str]]:
        return merge_validations([
            self.xtransformer.validate_input(df=df),
            self.ytransformer.validate_input(df=df),
        ])

    def check_df_train(self, df: DataFrame) -> Tuple[bool, List[str]]:
        validations = [self.check_df_prediction(df)]
        if self.wtransformer is not None:
            validations.append(self.wtransformer.validate_input(df=df))
        return merge_validations(validations)

    def fit(self, df: DataFrame) -> ETLWrapperForLGBM:
        ok, msgs = self.check_df_train(df)
        if not ok:
            raise ValueError(msgs)
        # get the matrices
        X = self.xtransformer.fit(df).get_transformed_as_pdf(df)
        y = self.ytransformer.fit(df).get_transformed_as_pdf(df)
        p = {}
        if self.wtransformer is not None:
            p['sample_weight'] = self.wtransformer.fit(df).get_transformed_as_pdf(df)
        # fit
        feature_name = self.xtransformer.feature_columns
        categorical_feature = self.xtransformer.categorical_feature_columns
        self.m.fit(X=X, y=y, feature_name=feature_name,
                   categorical_feature=categorical_feature, **p)
        return self

    def predict(self, df: DataFrame) -> DataFrame:
        ok, msgs = self.check_df_prediction(df)
        if not ok:
            raise ValueError(msgs)
        # ensure rows are identifiable
        jb = JoinableByRowID(df)
        # get the matrices
        Xpred = self.xtransformer.get_transformed_as_pdf(jb.df)
        # predict
        ypred = self.m.predict(Xpred)
        # join them back to df
        df_with_y = jb.join_by_row_id(ypred, column=self.ytransformer.feature_columns[0])
        df_pred = self.ytransformer.inverse_transform(df_with_y)
        return df_pred


def all_non_negative(df: DataFrame, c: str) -> bool:
    IS_NON_NEG = '__IS_NON_NEG__'
    n = df.count()
    df = (
        df
        .withColumn(IS_NON_NEG, F.col(c))
        .groupBy(IS_NON_NEG).count()
    )
    d = {
        row[IS_NON_NEG]: row['count']
        for row in df.collect()
    }
    return d[True] == n


def get_top_correlated(df: DataFrame, c: str) -> Tuple[Optional[str], Optional[float]]:
    ABS_VAL_COL = '__ABS_VAL_COL__'
    corr_df = correlation(
        df=df,
        feature_columns=[c for c in df.colums if c != c],
        target_column=c
    )
    corr_df.drop(c)
    top_corr_col, top_corr_val = None, None
    if len(corr_df) > 1:
        corr_df[ABS_VAL_COL] = corr_df[c].apply(lambda v: abs(v))
        corr_df = corr_df.sort_values(by=ABS_VAL_COL)
        top_corr_col, top_corr_val = corr_df.index.values[0], corr_df.iloc[0]
    return top_corr_col, top_corr_val


def recommend(df: DataFrame, target_column: str) -> ETLConfig:
    conf = ETLConfig(target_column=target_column)
    msgs = []
    # target_log_transform
    if all_non_negative:
        conf.target_log_transform = True
        msgs.append(
            'Setting target_log_transform to True '
            'since all target values are non negative.'
        )
    # TODO target_normalize_by_categorical
    # target_normalize_by_numerical
    top_corr_col, top_corr_val = get_top_correlated(df=df, c=target_column)
    if top_corr_col is not None and top_corr_val > 0.5:
        conf.target_normalize_by_numerical = top_corr_col
        msgs.append(
            f'Setting target_normalize_by_numerical to {top_corr_col} '
            f'since correlation with target {target_column} is {top_corr_val} (> 0.5)'
        )

    feature_cols = [c for c in df.columns if c != target_column]
    x_categorical_columns = []
    x_numeric_features = []
    weight_group_columns = []
    for c in feature_cols:
        dtype = df.schema[c].dataType
        if isinstance(dtype, DateType):
            conf.date_column = c
            msgs.append(f'Setting date_column to {c} as it is a DateType')
            weight_group_columns.append(c)
        elif isinstance(dtype, NumericType):
            x_numeric_features.append(c)
        elif isinstance(dtype, StringType):
            x_categorical_columns.append(c)
        else:
            msgs.append(f'Ignoring column {c}')
        feature_cols.pop(feature_cols.index(c))

    conf.x_categorical_columns = x_categorical_columns
    msgs.append(f'Setting x_categorical_columns as {x_categorical_columns}')
    conf.x_numeric_features = x_numeric_features
    msgs.append(f'Setting x_numeric_features as {x_numeric_features}')
    if len(weight_group_columns) > 0:
        conf.weight_group_columns = weight_group_columns
        msgs.append(f'Setting weight_group_columns as {weight_group_columns}')
    return conf
