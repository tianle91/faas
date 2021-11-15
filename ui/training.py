import logging
import os
from tempfile import TemporaryDirectory

import streamlit as st
from pyspark.sql import SparkSession

from faas.e2e import E2EPipline, plot_feature_importances
from faas.eda.iid import correlation, plot_target_correlation
from faas.storage import write_model
from faas.utils.dataframe import (get_date_columns, get_non_numeric_columns,
                                  get_numeric_columns)
from faas.utils.io import dump_file_to_location
from faas.utils.types import load_csv

logger = logging.getLogger(__name__)


def run_training():

    spark = SparkSession.builder.getOrCreate()

    st.title('Training')
    training_file = st.file_uploader('Training data', type='csv')

    with TemporaryDirectory() as temp_dir:
        if training_file is not None:

            # get the file into a local path
            training_path = os.path.join(temp_dir, 'train.csv')
            dump_file_to_location(training_file, p=training_path)
            df = load_csv(spark=spark, p=training_path)

            ########################################################################################
            st.header('What to train?')
            st.write(df.limit(10).toPandas())

            target_column = st.selectbox('target column', options=df.columns)

            _ = st.selectbox(label='date column', options=get_date_columns(df))

            categorical_columns = get_non_numeric_columns(df)
            if target_column in categorical_columns:
                categorical_columns.pop(categorical_columns.index(target_column))
            categorical_columns = st.multiselect(
                'categorical columns', options=categorical_columns, default=categorical_columns
            )

            numeric_columns = get_numeric_columns(df)
            if target_column in numeric_columns:
                numeric_columns.pop(numeric_columns.index(target_column))
            numeric_columns = st.multiselect(
                'numeric columns', options=numeric_columns, default=numeric_columns
            )

            with st.expander('Correlation'):
                if target_column in get_numeric_columns(df):
                    corr_df = correlation(
                        df,
                        feature_columns=numeric_columns,
                        target_column=target_column
                    )
                    st.pyplot(plot_target_correlation(corr_df, target_column=target_column))
                else:
                    st.warning(
                        'Correlation not available because '
                        f'target: {target_column} is not numeric.'
                    )

            ########################################################################################
            st.header('Train Now')
            train_now = st.select_slider('Train now?', options=['No', 'Yes'])
            if train_now == 'Yes':
                e2e = E2EPipline(
                    df=df,
                    target_column=target_column,
                    feature_columns=categorical_columns + numeric_columns,
                ).fit(df)
                st.session_state['trained_model'] = e2e
                key = write_model(e2e)
                st.success(f'Model key (save this for prediction): `{key}`')
                logger.info(f'wrote model key: {key}')

                with st.expander('Feature importances'):
                    st.pyplot(plot_feature_importances(m=e2e.m))
