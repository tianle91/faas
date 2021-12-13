import streamlit as st
from pyspark.sql.dataframe import DataFrame

from faas.utils.dataframe import filter_by_dict
from ui.config import Config
from ui.visualization.vis_iid import vis_ui_iid
from ui.visualization.vis_spatial import vis_ui_spatial
from ui.visualization.vis_ts import vis_ui_ts


def run_eda(conf: Config, df: DataFrame, st_container=None):
    if st_container is None:
        st_container = st

    st_container.header('Exploratory analysis')
    group = None
    if conf.group_columns is not None:
        group = st_container.selectbox(
            label='Plot group',
            options=[None, ] + conf.get_distinct_group_values(df=df),
            key='vis_ui_iid_plot_group'
        )
    vis_df_group = df
    if group is not None:
        vis_df_group = filter_by_dict(df=df, d=group).cache()

    st_container.markdown('## Scatter Plot')
    vis_ui_iid(df=vis_df_group, config=conf, st_container=st_container)
    if conf.date_column is not None:
        st_container.markdown('## Time Series Plot')
        vis_ui_ts(df=vis_df_group, config=conf, st_container=st_container)
    if conf.has_spatial_columns:
        st_container.markdown('## Spatial Plot')
        vis_ui_spatial(df=vis_df_group, config=conf, st_container=st_container)
