import streamlit as st

from faas.storage import list_models
import pprint as pp


def run_list():
    all_models = list_models()
    st.code(pp.pformat(all_models))
