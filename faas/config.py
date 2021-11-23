import logging
import pprint as pp
from dataclasses import dataclass
from typing import List, Optional, Tuple

import pyspark.sql.functions as F
from pyspark.sql import DataFrame
from pyspark.sql.types import DateType, NumericType, StringType

from faas.eda.iid import correlation

logger = logging.getLogger(__name__)


@dataclass
class FeatureConfig:
    categorical_columns: List[str]
    numeric_columns: List[str]
    date_column: Optional[str] = None


@dataclass
class TargetConfig:
    column: str
    log_transform: bool = False
    categorical_normalization_column: Optional[str] = None
    numerical_normalization_column: Optional[str] = None


@dataclass
class WeightConfig:
    date_column: Optional[str] = None
    annual_decay_rate: float = .1
    group_columns: Optional[List[str]] = None


@dataclass
class Config:
    feature: FeatureConfig
    target: TargetConfig
    weight: WeightConfig = WeightConfig()

    def to_dict(self):
        return {k: getattr(self, k).__dict__ for k in self.__dict__}


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


def recommend_config(df: DataFrame, target_column: str) -> Config:
    """Recommend Config to use.

    TODO: split into iid, ts, multi_ts

    Args:
        df (DataFrame): dataframe to run analysis on
        target_column (str): target column
    """
    # infer from column types
    date_column = None
    categorical_columns = []
    numeric_features = []
    for c in df.columns:
        if c != target_column:
            dtype = df.schema[c].dataType
            print(c, dtype)
            if isinstance(dtype, DateType):
                date_column = c
            elif isinstance(dtype, NumericType):
                numeric_features.append(c)
            elif isinstance(dtype, StringType):
                categorical_columns.append(c)

    # FeatureConfig
    feature = FeatureConfig(
        categorical_columns=categorical_columns,
        numeric_columns=numeric_features,
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
    weight = WeightConfig(date_column=date_column)

    return Config(feature=feature, target=target, weight=weight)
