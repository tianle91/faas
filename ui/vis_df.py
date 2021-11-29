import streamlit as st
from pyspark.sql import DataFrame


def preview_df(df: DataFrame, n: int = 100):
    prewview_pdf = df.limit(n).toPandas()
    st.markdown(f'Preview for first {n} rows out of {df.count()} loaded.')
    st.dataframe(prewview_pdf)
