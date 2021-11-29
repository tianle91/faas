import pyspark.sql.functions as F
from pyspark.sql.dataframe import DataFrame

from faas.config import Config
from faas.config.config import create_etl_config
from faas.transformer.lightgbm import ETLWrapperForLGBM


def get_trained(conf: Config, df: DataFrame) -> ETLWrapperForLGBM:
    if conf.date_column is not None:
        df = df.withColumn(
            conf.date_column, F.to_date(conf.date_column, conf.date_column_format))
    m = ETLWrapperForLGBM(config=create_etl_config(conf=conf, df=df))
    m.fit(df=df)
    return m


def get_prediction(conf: Config, df: DataFrame, m: ETLWrapperForLGBM) -> DataFrame:
    if conf.date_column is not None:
        df = df.withColumn(
            conf.date_column, F.to_date(conf.date_column, conf.date_column_format))

    ok, msgs = m.check_df_prediction(df=df)
    if not ok:
        raise ValueError('\n'.join(msgs))

    unused_columns = [c for c in df.columns if c not in conf.used_columns_prediction]
    if len(unused_columns) > 0:
        msgs.append(f'Unused columns: {unused_columns}')

    df_predict = m.predict(df)
    return df_predict, msgs
