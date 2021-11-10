import streamlit as st
from pyspark.sql import DataFrame

from faas.e2e import E2EPipline, check_categorical, check_numeric, check_target


def run_target_checklist(e2e: E2EPipline, df: DataFrame) -> bool:
    ok, messages = check_target(e2e, df=df)
    st.markdown('Target column: ' + '✅' if ok else '❌')
    if not ok:
        for message in messages:
            st.error(message)
    return ok


def run_numeric_features_checklist(e2e: E2EPipline, df: DataFrame) -> bool:
    ok, messages = check_numeric(e2e, df=df)
    if not ok:
        for message in messages:
            st.error(message)
    return ok


def run_categorical_features_checklist(e2e: E2EPipline, df: DataFrame) -> bool:
    ok, messages = check_categorical(e2e, df=df)
    if not ok:
        for message in messages:
            st.error(message)
    return ok


def run_features_checklist(e2e: E2EPipline, df: DataFrame) -> bool:
    numeric_ok = run_numeric_features_checklist(e2e, df=df)
    categorical_ok = run_categorical_features_checklist(e2e, df=df)
    ok = numeric_ok and categorical_ok
    st.markdown('All features: ' + '✅' if ok else '❌')
    if not ok:
        st.markdown('Numeric features: ' + '✅' if numeric_ok else '❌')
        st.markdown('Categorical features: ' + '✅' if categorical_ok else '❌')
    return ok
