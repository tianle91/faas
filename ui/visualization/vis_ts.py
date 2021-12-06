from typing import Optional

import plotly.express as px
import streamlit as st
from plotly.graph_objs._figure import Figure
from pyspark.sql import DataFrame

from faas.config import Config
from faas.utils.dataframe import filter_by_dict


def plot_ts(
    df: DataFrame,
    config: Config,
    group: Optional[dict] = None,
    color_feature: Optional[str] = None,
) -> Figure:
    if group is not None:
        df = filter_by_dict(df=df, d=group)

    # what do we need?
    select_cols = [config.date_column, config.target]
    p = {}
    if color_feature is not None:
        select_cols.append(color_feature)
        p['color'] = color_feature
    pdf = df.select(*select_cols).toPandas()
    fig = px.scatter(pdf, x=config.date_column, y=config.target, marginal_y='histogram', **p)
    return fig


def vis_ui_ts(df: DataFrame, config: Config):
    group = None
    if config.group_columns is not None:
        all_groups = [
            {k: row[k] for k in config.group_columns}
            for row in df.select(*config.group_columns).distinct().collect()
        ]
        group = st.selectbox(label='Plot group', options=[None, ] + all_groups)

    color_feature = st.selectbox(label='Color Feature', options=config.feature_columns)
    st.plotly_chart(plot_ts(df, config=config, group=group, color_feature=color_feature))
