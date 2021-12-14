from faas.config import Config
from faas.transformer.etl import WeightConfig


def create_weight_config(conf: Config) -> WeightConfig:
    group_columns = conf.group_columns
    if conf.target_is_categorical:
        if group_columns is None:
            group_columns = [conf.target]
        else:
            group_columns = [*group_columns, conf.target]
    return WeightConfig(
        date_column=conf.date_column,
        group_columns=group_columns
    )
