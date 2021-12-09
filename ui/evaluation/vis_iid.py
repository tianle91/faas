from typing import Optional

import plotly.express as px
import pyspark.sql.functions as F
import streamlit as st
from plotly.graph_objs._figure import Figure
from pyspark.sql import DataFrame

from faas.config import Config
from faas.utils.dataframe import filter_by_dict


def plot_evaluate_iid(
    df_predict: DataFrame,
    df_actual: DataFrame,
    config: Config,
    group: Optional[dict] = None,
    color_feature: Optional[str] = None
) -> Figure:
    if group is not None:
        df_predict = filter_by_dict(df=df_predict, d=group)
        df_actual = filter_by_dict(df=df_actual, d=group)

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

    select_cols = [PREDICTION_COL, ACTUAL_COL]
    if color_feature is not None:
        select_cols.append(color_feature)
    pdf = df_merged.select(*select_cols).toPandas()

    if config.target_is_categorical:
        fig = px.density_heatmap(pdf, x=ACTUAL_COL, y=PREDICTION_COL)
    else:
        fig = px.scatter(pdf, x=ACTUAL_COL, y=PREDICTION_COL, color=color_feature)
    return fig


def vis_evaluate_iid(df_predict: DataFrame, df_actual: DataFrame, config: Config):

    # color_feature is counts by default for categorical target
    color_feature = None
    if not config.target_is_categorical:
        color_feature = st.selectbox('Color Feature', options=config.feature_columns)

    group = None
    if config.group_columns is not None:
        all_groups = [
            {k: row[k] for k in config.group_columns}
            for row in df_actual.select(*config.group_columns).distinct().collect()
        ]
        group = st.selectbox(label='Plot group', options=[None, ] + all_groups)

    st.plotly_chart(plot_evaluate_iid(
        df_predict=df_predict,
        df_actual=df_actual,
        config=config,
        group=group,
        color_feature=color_feature
    ))
