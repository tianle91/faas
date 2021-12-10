import logging
import os
import pprint as pp
from tempfile import TemporaryDirectory

import pandas as pd
import streamlit as st
from pyspark.sql import DataFrame, SparkSession

from faas.config import Config
from faas.helper import get_prediction
from faas.storage import decrement_num_calls_remaining, read_model
from faas.utils.dataframe import has_duplicates
from faas.utils.io import load_cached_df_from_st_uploaded
from ui.visualization.vis_df import preview_df
from ui.visualization.vis_lightgbm import get_vis_lgbmwrapper

logger = logging.getLogger(__name__)

spark = (
    SparkSession
    .builder
    .appName('ui_predict')
    .config('spark.driver.maxResultsSize', '16g')
    .getOrCreate()
)


def highlight_target(s: pd.Series, target_column: str):
    if s.name == target_column:
        return ['background-color: #ff0000; color: white'] * len(s)
    else:
        return [''] * len(s)


def preview_prediction(df_predict: DataFrame, config: Config, n: int = 100):
    st.markdown(f'Preview for first {n}/{df_predict.count()} predictions.')

    preview_columns = [config.target]
    if config.date_column is not None:
        preview_columns.append(config.date_column)
    if config.has_spatial_columns:
        preview_columns += [config.latitude_column, config.longitude_column]
    if config.group_columns is not None:
        preview_columns += config.group_columns
    preview_columns += config.feature_columns

    preview_columns = list(set(preview_columns))

    pdf_predict = df_predict.limit(n).toPandas()
    st.dataframe(pdf_predict[preview_columns].style.apply(
        lambda s: highlight_target(s, target_column=config.target),
        axis=0
    ))


def run_predict(st_container=None):
    if st_container is None:
        st_container = st

    st_container.title('Predict')
    model_key = st.session_state.get('model_key', '')
    model_key = st_container.text_input('Model key (obtain this from training)', value=model_key)

    stored_model = None
    if model_key != '':
        try:
            stored_model = read_model(key=model_key)
            st.session_state['model_key'] = model_key
        except KeyError as e:
            st_container.error(e)

    if stored_model is not None:
        st_container.success('Model loaded!')
        with st_container.expander('Model visualization'):
            st_container.code(pp.pformat(stored_model.config.__dict__))
            get_vis_lgbmwrapper(stored_model.m)

        st_container.markdown('# Upload dataset')
        predict_file = st_container.file_uploader('Predict data', type=['csv', 'parquet'])

        if predict_file is not None:
            df = load_cached_df_from_st_uploaded(f=predict_file, spark=spark)
            st_container.header('Uploaded dataset')
            preview_df(df=df)

            if stored_model.num_calls_remaining <= 0:
                st_container.error(f'Num calls remaining: {stored_model.num_calls_remaining}')
            else:
                config = stored_model.config
                st_container.header('Prediction')

                with st_container.spinner('Running predictions...'):
                    df_predict, msgs = get_prediction(conf=config, df=df, m=stored_model.m)

                st_container.markdown('\n\n'.join([f'❌ {msg}' for msg in msgs]))

                if df_predict is not None:
                    df_predict = df_predict.cache()
                    logger.info(f'df_predict.columns: {df_predict.columns}')

                    # update num_calls_remaining
                    stored_model = decrement_num_calls_remaining(key=model_key)
                    st_container.info(f'Num calls remaining: {stored_model.num_calls_remaining}')

                    # evaluation
                    if config.target in df.columns:
                        st_container.success(
                            f'Detected target column: {config.target} in uploaded dataframe.')
                        if has_duplicates(df.select(config.used_columns_prediction)):
                            st_container.error(
                                'Cannot perform comparison as df has duplicates in '
                                f'used_columns_prediction: {config.used_columns_prediction}.'
                            )
                        else:
                            st_container.session_state['df_predict'] = df_predict
                            st_container.session_state['df_actual'] = df
                            st_container.success('Evaluation possible!')

                    preview_prediction(df_predict=df_predict, config=config)

                    if st_container.button('Export predictions'):
                        with st_container.spinner('Exporting predictions...'):
                            pdf_predict = df_predict.toPandas()
                        st_container.download_button(
                            f'Download all {df_predict.count()} predictions',
                            data=pdf_predict.to_csv(),
                            file_name='prediction.csv'
                        )
