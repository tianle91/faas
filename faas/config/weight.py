from faas.config import Config
from faas.transformer.etl import WeightConfig


def create_weight_config(conf: Config) -> WeightConfig:
    group_columns = conf.group_columns
    if conf.target_is_categorical:
        group_columns = [] if group_columns is None else group_columns
        group_columns.append(conf.target)
    return WeightConfig(
        date_column=conf.date_column,
        group_columns=group_columns
    )
