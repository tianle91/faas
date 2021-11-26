from faas.config import Config
from faas.transformer.etl import WeightConfig


def create_weight_config(conf: Config) -> WeightConfig:
    return WeightConfig(date_column=conf.date_column, group_columns=conf.group_columns)
