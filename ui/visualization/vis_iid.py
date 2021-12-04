from typing import Optional

import plotly.express as px
import streamlit as st
from plotly.graph_objs._figure import Figure
from pyspark.sql import DataFrame

from faas.config import Config
from faas.utils.dataframe import has_duplicates


def plot_iid(
    df: DataFrame,
    config: Config,
    x_axis_feature: Optional[str] = None,
    color_feature: Optional[str] = None,
) -> Figure:
    select_cols = [config.target]
    if x_axis_feature is not None:
        if x_axis_feature not in config.feature_columns:
            raise ValueError(
                f'horizontal_feature: {x_axis_feature} '
                f'not in config.feature_columns {config.feature_columns}'
            )
        select_cols.append(x_axis_feature)
    if color_feature is not None:
        if color_feature not in config.feature_columns:
            raise ValueError(
                f'color_feature: {color_feature} '
                f'not in config.feature_columns {config.feature_columns}'
            )
        select_cols.append(color_feature)

    # what do we need?
    pdf = df.select(*select_cols).toPandas()
    fig = px.scatter(pdf, x=x_axis_feature, y=config.target, color=color_feature)
    return fig


def vis_ui_iid(df: DataFrame, config: Config):
    horizontal_feature = st.selectbox('X-Axis Feature', options=config.feature_columns)

    color_feature = None
    other_possible_features = [c for c in config.feature_columns if c != horizontal_feature]
    if len(other_possible_features) > 0:
        color_feature = st.selectbox('Color Feature', options=other_possible_features)

    st.plotly_chart(plot_iid(
        df=df, config=config, x_axis_feature=horizontal_feature, color_feature=color_feature))


def plot_evaluate_iid(
    df_predict: DataFrame,
    df_actual: DataFrame,
    config: Config,
    color_feature: Optional[str] = None
):
    df_predict = df_predict.select(*config.feature_columns, config.target)
    if has_duplicates(df_predict):
        raise ValueError('Cannot evaluate as df_predict has duplicate feature columns.')
    df_actual = df_actual.select(*config.feature_columns, config.target)
    if has_duplicates(df_actual):
        raise ValueError('Cannot evaluate as df_actual has duplicate feature columns.')

    PREDICTION_COL = '__PREDICTION__'
    ACTUAL_COL = '__ACTUAL__'
    df_predict = df_predict.withColumnRenamed(config.target, PREDICTION_COL)
    df_actual = df_actual.withColumnRenamed(config.target, ACTUAL_COL)
    df_merged = df_actual.join(df_predict, on=config.feature_columns, how='left')
    print(df_merged.columns)

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
