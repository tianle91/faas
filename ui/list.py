import pprint as pp

import streamlit as st

from faas.storage import list_models
from ui.visualization.vis_lightgbm import get_vis_lgbmwrapper


def run_list(st_container=None):
    if st_container is None:
        st_container = st

    all_models = list_models()
    if len(all_models) == 0:
        st.warning('No trained models yet!')
    for model_key, stored_model in all_models.items():
        with st.expander(f'Model: {model_key} created at: {stored_model.dt}'):
            st.markdown(
                f'model_key: `{model_key}` '
                f'num_calls_remaining: {stored_model.num_calls_remaining}'
            )
            st.code(pp.pformat(stored_model.config.__dict__, compact=True))
            get_vis_lgbmwrapper(stored_model.m)
