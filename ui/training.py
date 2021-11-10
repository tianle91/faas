import os
import pickle
from tempfile import TemporaryDirectory

import pandas as pd
import streamlit as st
from pyspark.sql import SparkSession

from faas.e2e import E2EPipline, plot_feature_importances
from faas.eda import correlation, plot_target_correlation


def run_training():

    spark = SparkSession.builder.getOrCreate()

    st.title('Training')
    training_file = st.file_uploader('Training data', type='csv')

    with TemporaryDirectory() as temp_dir:
        if training_file is not None:
            # get the file into a local path
            training_path = os.path.join(temp_dir, 'train.csv')
            pd.read_csv(training_file).to_csv(training_path, index=False)

            df = spark.read.options(header=True, inferSchema=True).csv(training_path)

            st.markdown('# Uploaded dataset')
            st.write(df.limit(10).toPandas())

            st.markdown('# What to train?')
            target_column = st.selectbox('target column', options=df.columns)
            non_target_columns = df.columns
            if target_column is not None:
                non_target_columns.pop(non_target_columns.index(target_column))

            feature_columns = st.multiselect(
                'feature columns', options=non_target_columns, default=non_target_columns)

            if target_column is not None and feature_columns is not None:
                corr_df = correlation(df, feature_columns=feature_columns,
                                      target_column=target_column)
                st.pyplot(plot_target_correlation(corr_df, target_column=target_column))

            st.markdown('# Train Now?')
            e2e = None
            if st.button('Yes'):
                e2e = E2EPipline(
                    df=df,
                    target_column=target_column,
                ).fit(df)

            if e2e is not None:
                st.markdown('# Done!')
                st.download_button(
                    'Download trained', data=pickle.dumps(e2e), file_name='trained.model')
                st.pyplot(plot_feature_importances(m=e2e.m))
