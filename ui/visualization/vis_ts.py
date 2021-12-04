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
    comparison_feature: Optional[str] = None,
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

    pdf = df.orderBy(config.date_column).toPandas()
    p = {}
    if comparison_feature is not None:
        if comparison_feature not in config.feature_columns:
            raise KeyError(
                f'comparison_feature: {comparison_feature} not found in config.feature_columns')
        if comparison_feature not in df.columns:
            raise KeyError(f'comparison_feature: {comparison_feature} not found in df.columns')
        p['color'] = comparison_feature
    fig = px.scatter(pdf, x=config.date_column, y=config.target, marginal_y='histogram', **p)
    return fig


def vis_ui_ts(df: DataFrame, config: Config):
    st.header('Time Series Visualization')
    group = None
    if config.group_columns is not None:
        all_groups = [
            {k: row[k] for k in config.group_columns}
            for row in df.select(*config.group_columns).distinct().collect()
        ]
        group = st.selectbox(label='Plot group', options=[None, ] + all_groups)

    comparison_feature = st.selectbox(label='Comparison Feature', options=config.feature_columns)
    st.plotly_chart(plot_ts(df, config=config, group=group, comparison_feature=comparison_feature))
