import logging
import os
import pprint as pp
from tempfile import TemporaryDirectory

import pandas as pd
import streamlit as st
from pyspark.sql import SparkSession

from faas.helper import get_prediction
from faas.storage import read_model, set_num_calls_remaining
from faas.utils.io import dump_file_to_location
from faas.utils.types import load_csv
from ui.visualization.vis_df import preview_df
from ui.visualization.vis_lightgbm import get_vis_lgbmwrapper

logger = logging.getLogger(__name__)

spark = SparkSession.builder.appName('ui_predict').getOrCreate()


def highlight_target(s: pd.Series, target_column: str):
    if s.name == target_column:
        return ['background-color: #ff0000; color: white'] * len(s)
    else:
        return [''] * len(s)


def run_predict():

    st.title('Predict')
    model_key = st.session_state.get('model_key', '')
    model_key = st.text_input('Model key (obtain this from training)', value=model_key)

    stored_model = None
    if model_key != '':
        try:
            stored_model = read_model(key=model_key)
        except KeyError as e:
            st.error(e)

    if stored_model is not None:
        st.success('Model loaded!')
        with st.expander('Model visualization'):
            st.code(pp.pformat(stored_model.config.__dict__))
            get_vis_lgbmwrapper(stored_model.m)

        st.markdown('# Upload dataset')
        predict_file = st.file_uploader('Predict data', type='csv')

        if predict_file is not None:
            with TemporaryDirectory() as temp_dir:
                # get the file into a local path
                predict_path = os.path.join(temp_dir, 'predict.csv')
                dump_file_to_location(predict_file, p=predict_path)
                df = load_csv(spark=spark, p=predict_path)

                st.header('Uploaded dataset')
                preview_df(df=df)

                if stored_model.num_calls_remaining <= 0:
                    st.error(f'Num calls remaining: {stored_model.num_calls_remaining}')
                else:
                    if st.button('Predict'):
                        st.header('Prediction')
                        df_predict, msgs = get_prediction(
                            conf=stored_model.config, df=df, m=stored_model.m)

                        # errors?
                        st.markdown('\n\n'.join([f'❌ {msg}' for msg in msgs]))
                        if df_predict is not None:
                            pdf_predict = df_predict.toPandas()

                            # update num_calls_remaining
                            set_num_calls_remaining(
                                key=model_key, n=stored_model.num_calls_remaining - 1)
                            stored_model = read_model(key=model_key)
                            st.info(f'Num calls remaining: {stored_model.num_calls_remaining}')

                            # preview
                            preview_n = 100
                            st.markdown(
                                f'Preview for first {preview_n}/{len(pdf_predict)} predictions.')
                            preview_columns = [
                                stored_model.config.target, *stored_model.config.feature_columns]
                            st.dataframe(pdf_predict[preview_columns].style.apply(
                                lambda s: highlight_target(
                                    s, target_column=stored_model.config.target),
                                axis=0
                            ))

                            # download
                            st.download_button(
                                f'Download all {len(pdf_predict)} predictions',
                                data=pdf_predict.to_csv(),
                                file_name='prediction.csv'
                            )
