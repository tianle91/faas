from typing import Optional, Tuple

import pyspark.sql.functions as F
from pyspark.sql import DataFrame
from pyspark.sql.types import DateType, NumericType, StringType

from faas.eda.iid import correlation
from faas.wrapper import ETLConfig


def min_val(df: DataFrame, c: str) -> bool:
    return df.agg(F.min(c).alias('min')).collect()[0]['min']


def get_top_correlated(df: DataFrame, c: str) -> Tuple[Optional[str], Optional[float]]:
    ABS_VAL_COL = '__ABS_VAL_COL__'
    corr_df = correlation(
        df=df,
        feature_columns=[c for c in df.columns if c != c],
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
    msgs = []

    feature_cols = [c for c in df.columns if c != target_column]
    date_column = None
    x_categorical_columns = []
    x_numeric_features = []
    for c in feature_cols:
        dtype = df.schema[c].dataType
        print(c, dtype)
        if isinstance(dtype, DateType):
            date_column = c
        elif isinstance(dtype, NumericType):
            x_numeric_features.append(c)
        elif isinstance(dtype, StringType):
            x_categorical_columns.append(c)
        else:
            msgs.append(f'Ignoring column {c}')

    conf = ETLConfig(
        x_categorical_columns=x_categorical_columns,
        x_numeric_features=x_numeric_features,
        target_column=target_column,
    )
    msgs.append(f'Setting x_categorical_columns as {x_categorical_columns}')
    msgs.append(f'Setting x_numeric_features as {x_numeric_features}')
    weight_group_columns = []
    if date_column is not None:
        conf.date_column = date_column
        weight_group_columns.append(c)
        msgs.append(f'Setting date_column to {c} as it is a DateType')
    if len(weight_group_columns) > 0:
        conf.weight_group_columns = weight_group_columns
        msgs.append(f'Setting weight_group_columns as {weight_group_columns}')

    # target_log_transform
    min_target_val = min_val(df=df, c=target_column)
    if min_target_val >= 0.:
        conf.target_log_transform = True
        msgs.append(
            'Setting target_log_transform to True '
            f'since min_target_val = {min_target_val}'
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
    return conf, msgs
