from typing import List

import pandas as pd
import streamlit as st
from pyspark.sql import DataFrame


def preview_df(df: DataFrame, n: int = 100):
    preview_pdf = df.limit(n).toPandas()
    st.markdown(f'Preview for first {n} rows out of {df.count()} loaded.')
    st.dataframe(preview_pdf)


def highlight_columns(df: pd.DataFrame, columns: List[str]):
    def fn(s: pd.Series) -> List[str]:
        if s.name in columns:
            return ['background-color: #ff0000; color: white'] * len(s)
        else:
            return [''] * len(s)
    return df.style.apply(fn, axis=0)
