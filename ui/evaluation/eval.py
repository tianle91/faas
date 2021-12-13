import streamlit as st
from pyspark.sql.dataframe import DataFrame

from faas.utils.dataframe import filter_by_dict
from ui.config import Config
from ui.evaluation.vis_iid import vis_evaluate_iid
from ui.evaluation.vis_spatial import vis_evaluate_spatial
from ui.evaluation.vis_ts import vis_evaluate_ts


def run_eval(conf: Config, df: DataFrame):
    group = None
    if conf.group_columns is not None:
        group = st.selectbox(
            label='Plot group',
            options=[None, ] + conf.get_distinct_group_values(df=df),
            key='vis_ui_iid_plot_group'
        )
    eval_df_group = df
    if group is not None:
        eval_df_group = filter_by_dict(df=df, d=group).cache()

    st.markdown('## Scatter Plot')
    vis_evaluate_iid(df_evaluation=eval_df_group, config=conf)
    if conf.date_column is not None:
        st.markdown('## Time Series Plot')
        vis_evaluate_ts(df_evaluation=eval_df_group, config=conf)
    if conf.has_spatial_columns:
        st.markdown('## Spatial Plot')
        vis_evaluate_spatial(df_evaluation=eval_df_group, config=conf)
