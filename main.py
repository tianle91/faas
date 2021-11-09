import os
from tempfile import TemporaryDirectory

import lightgbm
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from pyspark.sql import SparkSession

from faas.e2e import E2EPipline
from faas.eda import correlation

spark = SparkSession.builder.getOrCreate()

st.markdown('# Training dataset')
training_file = st.file_uploader('Training data', type='csv')

with TemporaryDirectory() as temp_dir:
    if training_file is not None:
        # get the file into a local path
        training_path = os.path.join(temp_dir, 'train.csv')
        pd.read_csv(training_file).to_csv(training_path, index=False)

        df = spark.read.options(header=True, inferSchema=True).csv(training_path)

        st.markdown('## Uploaded dataset')
        st.write(df.limit(10).toPandas())

        st.markdown('## What to train?')
        target_column = st.selectbox('target column', options=df.columns)
        non_target_columns = df.columns
        if target_column is not None:
            non_target_columns.pop(non_target_columns.index(target_column))

        feature_columns = st.multiselect(
            'feature columns', options=non_target_columns, default=non_target_columns)

        if target_column is not None and feature_columns is not None:
            st.markdown('### Correlation')
            corr_df = correlation(df, feature_columns=feature_columns, target_column=target_column)
            st.write(corr_df)

        st.markdown('## Training')
        e2e = None
        if st.button('Train'):
            e2e = E2EPipline(
                df=df,
                target_column=target_column,
            ).fit(df)
            fig, ax = plt.subplots(figsize=(5, 5))
            lightgbm.plot_importance(e2e.m, ax=ax)
            st.pyplot(fig)
