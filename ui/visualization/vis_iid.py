from typing import Optional

import plotly.express as px
import streamlit as st
from plotly.graph_objs._figure import Figure
from pyspark.sql import DataFrame

from faas.config import Config


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
