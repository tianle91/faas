import pprint as pp

import streamlit as st
from pyspark.sql import DataFrame

from faas.storage import read_model
from ui.evaluation.utils import validate_evaluation
from ui.evaluation.vis_iid import vis_evaluate_iid
from ui.evaluation.vis_spatial import vis_evaluate_spatial
from ui.evaluation.vis_ts import vis_evaluate_ts
from ui.visualization.vis_lightgbm import get_vis_lgbmwrapper


def run_evaluation(st_container=None):
    if st_container is None:
        st_container = st

    st_container.title('Evaluation')

    df_predict: DataFrame = st.session_state.get('df_predict', None)
    df_actual: DataFrame = st.session_state.get('df_actual', None)

    if df_predict is None or df_actual is None:
        st.error('Upload dataframe with actuals and run predictions for evaluation.')
    else:
        model_key = st.session_state.get('model_key')
        stored_model = read_model(key=model_key)
        config = stored_model.config

        st_container.markdown(f'Target feature: `{config.target}`')

        validate_evaluation(df=df_predict, config=config)
        validate_evaluation(df=df_actual, config=config)

        with st_container.expander('Model visualization'):
            st_container.code(pp.pformat(stored_model.config.__dict__))
            get_vis_lgbmwrapper(stored_model.m)

        vis_evaluate_iid(df_predict=df_predict, df_actual=df_actual, config=config)

        if config.date_column is not None:
            with st_container.expander('Time Series Visualization'):
                vis_evaluate_ts(df_predict=df_predict, df_actual=df_actual, config=config)
        if config.has_spatial_columns:
            with st_container.expander('Spatial Visualization'):
                vis_evaluate_spatial(df_predict=df_predict, df_actual=df_actual, config=config)
