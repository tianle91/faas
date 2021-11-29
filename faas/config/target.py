import logging
from typing import Optional, Tuple

import pyspark.sql.functions as F
from pyspark.sql import DataFrame
from pyspark.sql.types import NumericType, StringType

from faas.config import Config
from faas.config.utils import get_columns_by_type
from faas.eda.iid import correlation
from faas.transformer.etl import TargetConfig

logger = logging.getLogger(__name__)


def min_val(df: DataFrame, c: str) -> bool:
    return df.agg(F.min(c).alias('min')).collect()[0]['min']


def get_top_correlated(df: DataFrame, c: str) -> Tuple[Optional[str], Optional[float]]:
    numeric_columns = get_columns_by_type(df=df, dtype=NumericType)
    categorical_columns = get_columns_by_type(df=df, dtype=StringType)
    corr_df = correlation(
        df=df,
        columns=numeric_columns + categorical_columns,
        collapse_categorical=True
    ).drop(c)[[c]]

    top_corr_col, top_corr_val = None, None
    if len(corr_df) > 1:
        ABS_VAL_COL = '__ABS_VAL_COL__'
        corr_df[ABS_VAL_COL] = corr_df[c].apply(lambda v: abs(v))
        corr_df = corr_df.sort_values(by=ABS_VAL_COL)
        top_corr_col = corr_df.index.values[0]
        top_corr_val = corr_df[ABS_VAL_COL].iloc[0]
    return top_corr_col, top_corr_val


def create_target_config(conf: Config, df: DataFrame) -> TargetConfig:
    target_p = {'column': conf.target}

    if conf.target_is_categorical:
        target_p['is_categorical'] = True
    else:
        min_target_val = min_val(df=df, c=conf.target)
        if min_target_val >= 0.:
            logger.info(
                f'Setting log_transform as True since min_target_val {min_target_val} >= 0.')
            target_p['log_transform'] = True
        # normalization
        top_corr_col, top_corr_val = get_top_correlated(df=df, c=conf.target)
        if top_corr_col is not None and top_corr_val > 0.5:
            top_corr_dtype = df.schema[top_corr_col].dataType
            if isinstance(top_corr_dtype, NumericType):
                target_p['numerical_normalization'] = top_corr_col
                logger.info(
                    f'Setting numerical_normalization to {top_corr_col} '
                    f'due to top_corr_val {top_corr_val} > 0.5.'
                )
            elif isinstance(top_corr_dtype, StringType):
                target_p['categorical_normalization_column'] = top_corr_col
                logger.info(
                    f'Setting categorical_normalization_column to {top_corr_col} '
                    f'due to top_corr_val {top_corr_val} > 0.5.'
                )
            else:
                raise TypeError(
                    f'Unexpected type for column: {top_corr_col} dtype: {top_corr_dtype}')
    return TargetConfig(**target_p)
