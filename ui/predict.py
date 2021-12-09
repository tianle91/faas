import logging
import os
import pprint as pp
from tempfile import TemporaryDirectory

import pandas as pd
import streamlit as st
from pyspark.sql import SparkSession

from faas.config import Config
from faas.helper import get_prediction
from faas.storage import decrement_num_calls_remaining, read_model
from faas.utils.dataframe import has_duplicates
from faas.utils.io import dump_file_to_location
from faas.utils.types import load_csv, load_parquet
from ui.visualization.vis_df import preview_df
from ui.visualization.vis_lightgbm import get_vis_lgbmwrapper

logger = logging.getLogger(__name__)

spark = SparkSession.builder.appName('ui_predict').getOrCreate()


def highlight_target(s: pd.Series, target_column: str):
    if s.name == target_column:
        return ['background-color: #ff0000; color: white'] * len(s)
    else:
        return [''] * len(s)


def preview_prediction(pdf_predict: pd.DataFrame, config: Config, n: int = 100):
    st.markdown(f'Preview for first {n}/{len(pdf_predict)} predictions.')

    preview_columns = [config.target]
    if config.date_column is not None:
        preview_columns.append(config.date_column)
    if config.has_spatial_columns:
        preview_columns += [config.latitude_column, config.longitude_column]
    if config.group_columns is not None:
        preview_columns += config.group_columns
    preview_columns += config.feature_columns

    preview_columns = list(set(preview_columns))

    st.dataframe(pdf_predict[preview_columns].style.apply(
        lambda s: highlight_target(s, target_column=config.target),
        axis=0
    ))


def run_predict():

    st.title('Predict')
    model_key = st.session_state.get('model_key', '')
    model_key = st.text_input('Model key (obtain this from training)', value=model_key)

    stored_model = None
    if model_key != '':
        try:
            stored_model = read_model(key=model_key)
            st.session_state['model_key'] = model_key
        except KeyError as e:
            st.error(e)

    if stored_model is not None:
        st.success('Model loaded!')
        with st.expander('Model visualization'):
            st.code(pp.pformat(stored_model.config.__dict__))
            get_vis_lgbmwrapper(stored_model.m)

        st.markdown('# Upload dataset')
        predict_file = st.file_uploader('Predict data', type=['csv', 'parquet'])

        if predict_file is not None:
            with TemporaryDirectory() as temp_dir:

                # load predict_file as spark dataframe
                if predict_file.name.endswith('.csv'):
                    training_path = os.path.join(temp_dir, 'predict.csv')
                    dump_file_to_location(predict_file, p=training_path)
                    df = load_csv(spark=spark, p=training_path)
                elif predict_file.name.endswith('.parquet'):
                    training_path = os.path.join(temp_dir, 'predict.parquet')
                    dump_file_to_location(predict_file, p=training_path)
                    df = load_parquet(spark=spark, p=training_path)

                # cache df in memory
                df.cache()
                df.count()

            st.header('Uploaded dataset')
            preview_df(df=df)

            if stored_model.num_calls_remaining <= 0:
                st.error(f'Num calls remaining: {stored_model.num_calls_remaining}')
            else:
                config = stored_model.config
                if st.button('Predict'):
                    st.header('Prediction')
                    df_predict, msgs = get_prediction(conf=config, df=df, m=stored_model.m)
                    st.markdown('\n\n'.join([f'❌ {msg}' for msg in msgs]))

                    if df_predict is not None:
                        df_predict = df_predict.cache()
                        logger.info(f'df_predict.columns: {df_predict.columns}')

                        # update num_calls_remaining
                        stored_model = decrement_num_calls_remaining(key=model_key)
                        st.info(f'Num calls remaining: {stored_model.num_calls_remaining}')

                        # preview and download
                        pdf_predict = df_predict.toPandas()
                        with st.expander('Preview'):
                            preview_prediction(pdf_predict=pdf_predict, config=config)
                        st.download_button(
                            f'Download all {len(pdf_predict)} predictions',
                            data=pdf_predict.to_csv(),
                            file_name='prediction.csv'
                        )

                        # evaluation
                        if config.target in df.columns:
                            st.success(
                                f'Detected target column: {config.target} in uploaded dataframe.')
                            if has_duplicates(df.select(config.used_columns_prediction)):
                                st.error(
                                    'Cannot perform comparison as df has duplicates in '
                                    f'used_columns_prediction: {config.used_columns_prediction}.'
                                )
                            else:
                                st.session_state['df_predict'] = df_predict
                                st.session_state['df_actual'] = df
                                st.success('Evaluation possible!')
