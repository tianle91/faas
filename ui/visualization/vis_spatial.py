from typing import Optional

import plotly.express as px
import streamlit as st
from plotly.graph_objs._figure import Figure
from pyspark.sql import DataFrame

from faas.config import Config


def plot_spatial(
    df: DataFrame,
    config: Config,
    location_name_column: Optional[str] = None
) -> Figure:
    if not config.has_spatial_columns:
        raise ValueError('Cannot plot spatial if not config.has_spatial_columns')

    select_cols = [
        config.latitude_column,
        config.longitude_column,
        config.target,
        *config.feature_columns
    ]
    if location_name_column is not None:
        select_cols.append(location_name_column)
    pdf = df.select(*select_cols).toPandas()
    fig = px.scatter_geo(
        pdf,
        lat=config.latitude_column,
        lon=config.longitude_column,
        color=config.target,
        hover_name=location_name_column,
        hover_data=[config.target, *config.feature_columns],
    )
    fig.update_layout(height=400, margin={"r": 0, "t": 0, "l": 0, "b": 0})
    return fig


def vis_ui_spatial(df: DataFrame, config: Config):
    location_name_column = st.selectbox('Location name column', options=[None] + df.columns)
    st.plotly_chart(plot_spatial(
        df=df,
        config=config,
        location_name_column=location_name_column
    ))
