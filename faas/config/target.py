import logging
from typing import Optional, Tuple

import pyspark.sql.functions as F
from pyspark.sql import DataFrame

from faas.config import Config
from faas.eda.iid import correlation
from faas.transformer.etl import TargetConfig

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


def create_target_config(conf: Config, df: DataFrame) -> TargetConfig:
    target_p = {'column': conf.target}

    min_target_val = min_val(df=df, c=conf.target)
    if min_target_val >= 0.:
        logger.info(f'Setting log_transform as True since min_target_val {min_target_val} >= 0.')
        target_p['log_transform'] = True

    top_corr_col, top_corr_val = get_top_correlated(df=df, c=conf.target)
    if top_corr_col is not None and top_corr_val > 0.5:
        target_p['numerical_normalization'] = top_corr_col
        logger.info(
            f'Setting numerical_normalization to {top_corr_col} '
            f'due to top_corr_val {top_corr_val} > 0.5.'
        )
    return TargetConfig(**target_p)