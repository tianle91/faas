import streamlit as st
from pyspark.sql import DataFrame

from faas.e2e import E2EPipline


def run_target_checklist(e2e: E2EPipline, df: DataFrame) -> bool:
    ok, messages = e2e.check_target(df)
    st.markdown('Target column: ' + '✅' if ok else '❌')
    if not ok:
        for message in messages:
            st.error(message)
    return ok


def run_numeric_features_checklist(e2e: E2EPipline, df: DataFrame) -> bool:
    ok, messages = e2e.check_numeric(df)
    if not ok:
        for message in messages:
            st.error(message)
    return ok


def run_categorical_features_checklist(e2e: E2EPipline, df: DataFrame) -> bool:
    ok, messages = e2e.check_categorical(df)
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
