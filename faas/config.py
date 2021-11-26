import logging
import pprint as pp
from typing import List, Optional, Tuple

import pyspark.sql.functions as F
from pyspark.sql import DataFrame
from pyspark.sql.types import DataType, NumericType, StringType

from faas.eda.iid import correlation
from faas.etl import Config, FeatureConfig, TargetConfig, WeightConfig

logger = logging.getLogger(__name__)


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


def get_columns_by_type(df: DataFrame, dtype: DataType) -> List[str]:
    out = []
    for c in df.columns:
        if isinstance(df.schema[c].dataType, dtype):
            out.append(c)
    return out


def recommend_config(
    df: DataFrame,
    target_column: str,
    date_column: Optional[str] = None,
    group_columns: Optional[List[str]] = None
) -> Config:
    """Recommend Config to use.

    TODO: split into iid, ts, multi_ts

    Args:
        df: dataframe to run analysis on
        target_column: target column
        date_column: date column, defaults to None.
        group_columns: group columns if date column is specified, defaults to None.
    """
    # infer from column types
    categorical_columns = get_columns_by_type(df=df, dtype=StringType)
    numeric_columns = get_columns_by_type(df=df, dtype=NumericType)
    if target_column not in numeric_columns:
        raise TypeError(f'target_column: {target_column} should be numeric')

    # FeatureConfig
    feature = FeatureConfig(
        categorical_columns=categorical_columns,
        numeric_columns=[c for c in numeric_columns if c != target_column],
        date_column=date_column
    )
    logger.info(f'Setting FeatureConfig: {pp.pformat(feature.__dict__)}')

    # TargetConfig
    target_p = {'column': target_column}
    # log_transform
    min_target_val = min_val(df=df, c=target_column)
    if min_target_val >= 0.:
        logger.info(f'Setting log_transform as True since min_target_val {min_target_val} >= 0.')
        target_p['log_transform'] = True
    # numerical_normalization
    top_corr_col, top_corr_val = get_top_correlated(df=df, c=target_column)
    if top_corr_col is not None and top_corr_val > 0.5:
        target_p['numerical_normalization'] = top_corr_col
        logger.info(
            f'Setting numerical_normalization to {top_corr_col} '
            f'due to top_corr_val {top_corr_val} > 0.5.'
        )
    target = TargetConfig(**target_p)
    logger.info(f'Setting TargetConfig: {pp.pformat(target.__dict__)}')

    # WeightConfig
    weight = WeightConfig(date_column=date_column, group_columns=group_columns)

    return Config(feature=feature, target=target, weight=weight)
