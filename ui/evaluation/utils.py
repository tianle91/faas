from pyspark.sql import DataFrame

from faas.config import Config
from faas.utils.dataframe import has_duplicates


def validate_evaluation(df: DataFrame, config: Config):
    config.validate_df(df=df)
    if has_duplicates(df.select(*config.used_columns_prediction)):
        raise ValueError('Cannot evaluate as df_predict has duplicate feature columns.')
