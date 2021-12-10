from typing import Optional

import plotly.express as px
import pyspark.sql.functions as F
import streamlit as st
from plotly.graph_objs._figure import Figure
from pyspark.sql import DataFrame

from faas.config import Config
from faas.utils.dataframe import filter_by_dict
from ui.predict import PREDICTION_COLUMN

IS_PREDICTION_COLUMN = '__IS_PREDICTION__'


def plot_ts(
    df_evaluation: DataFrame,
    config: Config,
    group: Optional[dict] = None
) -> Figure:
    if group is not None:
        df_evaluation = filter_by_dict(df=df_evaluation, d=group)

    actual_df = (
        df_evaluation
        .withColumn(IS_PREDICTION_COLUMN, F.lit(False))
        .select(config.date_column, config.target)
    )
    pred_df = (
        df_evaluation
        .withColumn(config.target, F.col(PREDICTION_COLUMN))
        .withColumn(IS_PREDICTION_COLUMN, F.lit(True))
        .select(config.date_column, config.target)
    )

    # what do we need?
    pdf = actual_df.union(pred_df).toPandas()
    fig = px.scatter(
        pdf,
        x=config.date_column,
        y=config.target,
        marginal_y='histogram',
        color=IS_PREDICTION_COLUMN
    )
    return fig


def vis_evaluate_ts(df_evaluation: DataFrame, config: Config):
    group = None
    if config.group_columns is not None:
        group = st.selectbox(
            label='Plot group',
            options=[None, ] + config.get_distinct_group_values(df=df_evaluation),
            key='vis_evaluate_ts_plot_group'
        )

    st.plotly_chart(plot_ts(
        df_evaluation=df_evaluation,
        config=config,
        group=group
    ))
