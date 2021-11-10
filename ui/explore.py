import streamlit as st

from faas.storage import list_models


def run_explore():
    all_models = list_models()
    for k, v in all_models.items():
        with st.expander(k):
            st.write(v)
