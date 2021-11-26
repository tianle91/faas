import pprint as pp

import streamlit as st

from faas.storage import list_models
from ui.vis_lightgbm import get_vis_lgbmwrapper


def run_list():
    all_models = list_models()
    for k in all_models:
        model_key, dt = k
        with st.expander(f'Model: {model_key} created at: {dt}'):
            st.code(model_key)
            m, conf = all_models[k]
            st.code(pp.pformat(conf.__dict__, compact=True))
            st.code(pp.pformat(m.config.to_dict(), compact=True))
            get_vis_lgbmwrapper(m)
