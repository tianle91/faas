from pyspark.sql import DataFrame
from pyspark.sql.types import NumericType, StringType

from faas.config import Config
from faas.config.utils import get_columns_by_type
from faas.transformer.etl import FeatureConfig


def create_feature_config(conf: Config, df: DataFrame) -> FeatureConfig:
    df = df.select(*conf.feature_columns)
    categorical_columns = get_columns_by_type(df=df, dtype=StringType)
    numeric_columns = get_columns_by_type(df=df, dtype=NumericType)
    p = {}
    if conf.date_column is not None:
        p['date_column'] = conf.date_column
    if conf.latitude_column is not None and conf.longitude_column is not None:
        p['latitude_column'] = conf.latitude_column
        p['longitude_column'] = conf.longitude_column
    return FeatureConfig(
        categorical_columns=categorical_columns,
        numeric_columns=numeric_columns,
        **p
    )
