from typing import Optional

import plotly.express as px
import pyspark.sql.functions as F
import streamlit as st
from plotly.graph_objs._figure import Figure
from pyspark.sql import DataFrame
from pyspark.sql.types import BooleanType, DoubleType

from faas.config import Config


def categorical_error(
    df: DataFrame, actual_col: str, predict_col: str, error_col: str = '__ERROR_'
) -> DataFrame:
    udf = F.udf(lambda a, b: a == b, BooleanType())
    return df.withColumn(
        error_col,
        udf(F.col(actual_col), F.col(predict_col))
    )


def numeric_error(
    df: DataFrame, actual_col: str, predict_col: str, error_col: str = '__ERROR_'
) -> DataFrame:
    udf = F.udf(lambda actual, predict: (predict - actual) / actual, DoubleType())
    return df.withColumn(
        error_col,
        udf(F.col(actual_col), F.col(predict_col))
    )


def plot_spatial(
    df_predict: DataFrame,
    df_actual: DataFrame,
    config: Config,
    location_name_column: Optional[str] = None
) -> Figure:
    PREDICTION_COL = '__PREDICTION__'
    ACTUAL_COL = '__ACTUAL__'
    df_predict = (
        df_predict
        .withColumn(PREDICTION_COL, F.col(config.target))
        .drop(config.target)
    )
    df_actual = (
        df_actual
        .withColumn(ACTUAL_COL, F.col(config.target))
        .drop(config.target)
    )
    df_merged = df_actual.join(df_predict, on=config.used_columns_prediction, how='left')

    ERROR_COL = '__ERROR__'
    if config.target_is_categorical:
        df_merged = categorical_error(
            df=df_merged,
            actual_col=ACTUAL_COL,
            predict_col=PREDICTION_COL,
            error_col=ERROR_COL
        )
    else:
        df_merged = numeric_error(
            df=df_merged,
            actual_col=ACTUAL_COL,
            predict_col=PREDICTION_COL,
            error_col=ERROR_COL
        )

    select_cols = config.used_columns_prediction + [PREDICTION_COL, ACTUAL_COL, ERROR_COL]
    if location_name_column is not None and location_name_column not in select_cols:
        select_cols.append(location_name_column)

    pdf = df_merged.select(*select_cols).toPandas()
    fig = px.scatter_geo(
        pdf,
        lat=config.latitude_column,
        lon=config.longitude_column,
        color=ERROR_COL,
        hover_name=location_name_column,
        hover_data=[PREDICTION_COL, ACTUAL_COL, ERROR_COL],
        fitbounds='locations'
    )
    return fig


def vis_evaluate_spatial(df_predict: DataFrame, df_actual: DataFrame, config: Config):
    location_name_column = st.selectbox(
        'Location name column',
        options=[None] + sorted(df_actual.columns)
    )
    st.plotly_chart(plot_spatial(
        df_predict=df_predict,
        df_actual=df_actual,
        config=config,
        location_name_column=location_name_column
    ))
