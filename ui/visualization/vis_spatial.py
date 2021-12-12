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
    select_cols = config.used_columns_prediction + [config.target]
    if location_name_column is not None and location_name_column not in select_cols:
        select_cols.append(location_name_column)
    pdf = df.select(*select_cols).toPandas()
    fig = px.scatter_geo(
        pdf,
        lat=config.latitude_column,
        lon=config.longitude_column,
        color=config.target,
        hover_name=location_name_column,
        hover_data=[config.target, *config.feature_columns],
        fitbounds='locations'
    )
    return fig


def vis_ui_spatial(df: DataFrame, config: Config, st_container=None):
    if st_container is None:
        st_container = st

    location_name_column = st_container.selectbox(
        'Location name column',
        options=[None] + sorted(df.columns)
    )
    st_container.plotly_chart(plot_spatial(
        df=df,
        config=config,
        location_name_column=location_name_column
    ))
