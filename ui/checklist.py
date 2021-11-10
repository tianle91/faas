import streamlit as st
from pyspark.sql import DataFrame
from pyspark.sql.types import NumericType

from faas.e2e import E2EPipline


def run_target_checklist(e2e: E2EPipline, df: DataFrame) -> bool:
    target_dtype = df.schema[e2e.target_column].dataType
    target_column_passed = (e2e.target_is_numeric == isinstance(target_dtype, NumericType))
    st.markdown('Target column: ' + '✅' if target_column_passed else '❌')
    if not target_column_passed:
        st.error(
            f'Expected target_is_numeric: {e2e.target_is_numeric} '
            f'but received {target_dtype} instead.'
        )
    return target_column_passed


def run_numeric_features_checklist(e2e: E2EPipline, df: DataFrame) -> bool:
    passed = {
        c: isinstance(df.schema[c].dataType, NumericType)
        for c in e2e.numeric_features
    }
    all_good = all(passed.values())
    if all_good:
        st.markdown('Numeric features: ' + '✅' if all_good else '❌')
    else:
        for c, v in passed.items():
            if not v:
                st.error(
                    f'Expected {c} to be numeric '
                    f'but received {df.schema[c].dataType} instead.'
                )
    return all_good


def run_categorical_features_checklist(e2e: E2EPipline, df: DataFrame) -> bool:
    passed = {
        c: not isinstance(df.schema[c].dataType, NumericType)
        for c in e2e.categorical_features
    }
    all_good = all(passed.values())
    if all_good:
        st.markdown('Categorical features: ' + '✅' if all_good else '❌')
    else:
        for c, v in passed.items():
            if not v:
                st.error(
                    f'Expected {c} to be not numeric '
                    f'but received {df.schema[c].dataType} instead.'
                )
    return all_good


def run_features_checklist(e2e: E2EPipline, df: DataFrame) -> bool:
    numeric_good = run_numeric_features_checklist(e2e, df=df)
    categorical_good = run_categorical_features_checklist(e2e, df=df)
    all_good = numeric_good and categorical_good
    st.markdown('All features: ' + '✅' if all_good else '❌')
    return all_good
