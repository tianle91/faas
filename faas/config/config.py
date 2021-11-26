from pyspark.sql import DataFrame

from faas.config import Config
from faas.config.feature import create_feature_config
from faas.config.target import create_target_config
from faas.config.weight import create_weight_config
from faas.transformer.etl import ETLConfig


def create_etl_config(conf: Config, df: DataFrame) -> ETLConfig:
    conf.validate()
    return ETLConfig(
        feature=create_feature_config(conf=conf, df=df),
        target=create_target_config(conf=conf, df=df),
        weight=create_weight_config(conf=conf)
    )
