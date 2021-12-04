import streamlit as st
from pyspark.sql import DataFrame

from faas.config import Config
from ui.visualization.vis_iid import vis_evaluate_iid


def get_evaluation(df_predict: DataFrame, df_actual: DataFrame, config: Config):
    st.header('Evaluation')
    with st.expander('Visualization'):
        vis_evaluate_iid(df_predict=df_predict, df_actual=df_actual, config=config)
