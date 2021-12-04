from typing import Optional

import plotly.express as px
import pyspark.sql.functions as F
import streamlit as st
from plotly.graph_objs._figure import Figure
from pyspark.sql import DataFrame

from faas.config import Config


def plot_ts(
    df: DataFrame,
    config: Config,
    group: Optional[dict] = None,
    color_feature: Optional[str] = None,
) -> Figure:
    if config.date_column is None:
        raise ValueError('Cannot plot time series if config.date_column is None.')
    if group is not None:
        if config.group_columns is not None:
            if set(group.keys()) == set(config.group_columns):
                for col, val in group.items():
                    df = df.filter(F.col(col) == val)
            else:
                raise ValueError(
                    f'Provided group: {group} '
                    f'must match config.group_columns: {config.group_columns}'
                )
        else:
            raise ValueError(
                f'group: {group} specified but no group specified in config.group_columns')

    # what do we need?
    select_cols = [config.date_column, config.target]
    plot_params = {}
    if color_feature is not None:
        if color_feature not in config.feature_columns:
            raise KeyError(
                f'comparison_feature: {color_feature} not found in config.feature_columns')
        if color_feature not in df.columns:
            raise KeyError(f'comparison_feature: {color_feature} not found in df.columns')
        select_cols.append(color_feature)
        plot_params['color'] = color_feature
    pdf = (
        df
        .select(*select_cols)
        .orderBy(config.date_column)
        .toPandas()
    )
    fig = px.scatter(
        pdf, x=config.date_column, y=config.target, marginal_y='histogram', **plot_params)
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
