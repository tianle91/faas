from typing import Optional

import plotly.express as px
import streamlit as st
from plotly.graph_objs._figure import Figure
from pyspark.sql import DataFrame

from faas.config import Config


def plot_ts(
    df: DataFrame,
    config: Config,
    color_feature: Optional[str] = None,
) -> Figure:
    select_cols = [config.date_column, config.target]
    p = {}
    if color_feature is not None:
        select_cols.append(color_feature)
        p['color'] = color_feature
    pdf = df.select(*select_cols).toPandas()
    fig = px.scatter(pdf, x=config.date_column, y=config.target, marginal_y='histogram', **p)
    return fig


def vis_ui_ts(df: DataFrame, config: Config):
    color_feature = st.selectbox(
        label='Color Feature',
        options=sorted(config.feature_columns)
    )
    st.plotly_chart(plot_ts(df, config=config, color_feature=color_feature))
