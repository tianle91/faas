from typing import Optional

import plotly.express as px
import pyspark.sql.functions as F
import streamlit as st
from plotly.graph_objs._figure import Figure
from pyspark.sql import DataFrame

from faas.config import Config
from faas.utils.dataframe import filter_by_dict


def plot_ts(
    df_predict: DataFrame,
    df_actual: DataFrame,
    config: Config,
    group: Optional[dict] = None
) -> Figure:
    if group is not None:
        df_predict = filter_by_dict(df=df_predict, d=group)
        df_actual = filter_by_dict(df=df_actual, d=group)

    IS_PREDICTION_COL = '__IS_PREDICTION__'
    df_predict = df_predict.withColumn(IS_PREDICTION_COL, F.lit(True))
    df_actual = df_actual.withColumn(IS_PREDICTION_COL, F.lit(False))
    df_merged = df_actual.union(df_predict)

    # what do we need?
    select_cols = [config.date_column, config.target, IS_PREDICTION_COL]
    pdf = df_merged.select(*select_cols).toPandas()
    fig = px.scatter(
        pdf, x=config.date_column, y=config.target, marginal_y='histogram', color=IS_PREDICTION_COL)
    return fig


def vis_evaluate_ts(df_predict: DataFrame, df_actual: DataFrame, config: Config):
    group = None
    if config.group_columns is not None:
        all_groups = [
            {k: row[k] for k in config.group_columns}
            for row in df_actual.select(*config.group_columns).distinct().collect()
        ]
        group = st.selectbox(label='Plot group', options=[None, ] + sorted(all_groups))

    st.plotly_chart(plot_ts(
        df_predict=df_predict,
        df_actual=df_actual,
        config=config,
        group=group
    ))
