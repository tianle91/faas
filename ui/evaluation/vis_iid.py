from typing import Optional

import plotly.express as px
import streamlit as st
from plotly.graph_objs._figure import Figure
from pyspark.sql import DataFrame

from faas.config import Config
from faas.utils.dataframe import filter_by_dict
from ui.predict import PREDICTION_COLUMN


def plot_evaluate_iid(
    df_evaluation: DataFrame,
    config: Config,
    group: Optional[dict] = None,
    color_feature: Optional[str] = None
) -> Figure:
    if group is not None:
        df_evaluation = filter_by_dict(df=df_evaluation, d=group)

    select_cols = [PREDICTION_COLUMN, config.target]
    if color_feature is not None:
        select_cols.append(color_feature)
    pdf = df_evaluation.select(*select_cols).toPandas()

    if config.target_is_categorical:
        fig = px.density_heatmap(pdf, x=config.target, y=PREDICTION_COLUMN)
    else:
        fig = px.scatter(pdf, x=config.target, y=PREDICTION_COLUMN, color=color_feature)
    return fig


def vis_evaluate_iid(df_evaluation: DataFrame, config: Config, st_container=None):
    if st_container is None:
        st_container = st

    # color_feature is counts by default for categorical target
    color_feature = None
    if not config.target_is_categorical:
        color_feature = st_container.selectbox(
            'Color Feature', options=sorted(config.feature_columns))

    group = None
    if config.group_columns is not None:
        group = st_container.selectbox(
            label='Plot group',
            options=[None, ] + config.get_distinct_group_values(df=df_evaluation),
            key='vis_evaluate_iid_plot_group'
        )

    st_container.plotly_chart(plot_evaluate_iid(
        df_evaluation=df_evaluation,
        config=config,
        group=group,
        color_feature=color_feature
    ))
