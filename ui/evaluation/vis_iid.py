from typing import Optional

import plotly.express as px
import pyspark.sql.functions as F
import streamlit as st
from plotly.graph_objs._figure import Figure
from pyspark.sql import DataFrame

from faas.config import Config


def plot_evaluate_iid(
    df_predict: DataFrame,
    df_actual: DataFrame,
    config: Config,
    color_feature: Optional[str] = None
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
    df_merged = df_actual.join(df_predict, on=config.feature_columns, how='left')

    select_cols = [PREDICTION_COL, ACTUAL_COL]
    if color_feature is not None:
        select_cols.append(color_feature)
    pdf = df_merged.select(*select_cols).toPandas()
    fig = px.scatter(pdf, x=ACTUAL_COL, y=PREDICTION_COL, color=color_feature)
    return fig


def vis_evaluate_iid(df_predict: DataFrame, df_actual: DataFrame, config: Config):
    color_feature = st.selectbox('Color Feature', options=config.feature_columns)
    st.plotly_chart(plot_evaluate_iid(
        df_predict=df_predict,
        df_actual=df_actual,
        config=config,
        color_feature=color_feature
    ))
