import pprint as pp

import streamlit as st

from faas.storage import list_models


def run_list(st_container=None):
    if st_container is None:
        st_container = st

    all_models = [(model_key, stored_model) for model_key, stored_model in list_models().items()]
    all_models.sort(key=lambda v: v[1].dt, reverse=True)

    if len(all_models) == 0:
        st_container.warning('No trained models yet!')

    for model_key, stored_model in all_models:
        st_container.markdown(
            f'''
            ----
            Model key: `{model_key}`

            Created at: `{stored_model.dt}`

            Number of predictions remaining: `{stored_model.num_calls_remaining}`
            '''
        )
        st_container.code(pp.pformat(stored_model.config.__dict__, compact=True))
