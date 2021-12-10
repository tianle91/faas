import streamlit as st
from pyspark.sql import DataFrame


def preview_df(df: DataFrame, n: int = 100, st_container=None):
    if st_container is None:
        st_container = st

    preview_pdf = df.limit(n).toPandas()
    st_container.markdown(f'Preview for first {n} rows out of {df.count()} loaded.')
    st_container.dataframe(preview_pdf)
