import streamlit as st

from ui.evaluation.utils import validate_evaluation
from ui.evaluation.vis_iid import vis_evaluate_iid
from ui.evaluation.vis_spatial import vis_evaluate_spatial
from ui.evaluation.vis_ts import vis_evaluate_ts
from ui.visualization.vis_lightgbm import get_vis_lgbmwrapper


def run_evaluation(st_container=None):
    if st_container is None:
        st_container = st

    st_container.title('Evaluation')

    df_evaluation = st.session_state.get('df_evaluation', None)
    stored_model = st.session_state.get('stored_model', None)
    if df_evaluation is None:
        st_container.error('Upload dataframe with actuals and run predictions for evaluation.')
        return None
    if stored_model is None:
        st_container.error('No stored model, please provide a model key.')

    config = stored_model.config
    st_container.markdown(f'Target feature: `{config.target}`')
    get_vis_lgbmwrapper(stored_model.m, st_container=st_container)

    validate_evaluation(df=df_evaluation, config=config)

    vis_evaluate_iid(df_evaluation=df_evaluation, config=config)
    if config.date_column is not None:
        with st_container.expander('Time Series Visualization'):
            vis_evaluate_ts(df_evaluation=df_evaluation, config=config)
    if config.has_spatial_columns:
        with st_container.expander('Spatial Visualization'):
            vis_evaluate_spatial(df_evaluation=df_evaluation, config=config)
