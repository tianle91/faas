import json
import os
import pprint as pp
from tempfile import TemporaryDirectory

import pandas as pd
import requests
import streamlit as st
from pyspark.sql import SparkSession

from faas.lightgbm import LGBMWrapper
from faas.storage import read_model
from faas.utils.io import dump_file_to_location

API_URL = 'http://localhost:8000'


def highlight_target(s: pd.Series, target_column: str):
    if s.name == target_column:
        return ['background-color: #ff0000; color: white'] * len(s)
    else:
        return ['background-color: #ffffff'] * len(s)


def run_predict():

    spark = SparkSession.builder.getOrCreate()

    st.title('Predict')
    model_key = st.session_state.get('model_key', '')
    model_key = st.text_input('Model key (obtain this from training)', value=model_key)

    m = None
    if model_key != '':
        try:
            m: LGBMWrapper = read_model(key=model_key)
        except KeyError as e:
            st.error(e)

    if m is not None:
        st.success('Model loaded!')
        with st.expander('Details on loaded model'):
            st.code(pp.pformat(m.config.to_dict()))

        st.markdown('# Upload dataset')
        predict_file = st.file_uploader('Predict data', type='csv')

        if predict_file is not None:
            with TemporaryDirectory() as temp_dir:
                # get the file into a local path
                predict_path = os.path.join(temp_dir, 'predict.csv')
                dump_file_to_location(predict_file, p=predict_path)

                # load the csv and send to api
                pred_pdf = pd.read_csv(predict_path)
                r = requests.post(
                    url=f'{API_URL}/predict/{model_key}',
                    data=json.dumps({'data': pred_pdf.to_dict(orient='records')})
                )
                response_json = r.json()

                if response_json['prediction'] is not None:
                    pred_pdf_received = pd.DataFrame(response_json['prediction'])
                    pred_pdf_preview = pred_pdf_received[[
                        *m.config.feature.categorical_columns,
                        *m.config.feature.numeric_columns,
                        m.config.target.column
                    ]].head(10)
                    st.dataframe(pred_pdf_preview.style.apply(
                        lambda s: highlight_target(s, target_column=m.config.target.column),
                        axis=0
                    ))
                    st.download_button(
                        'Download prediction',
                        data=pred_pdf_received.to_csv(),
                        file_name='prediction.csv'
                    )
                else:
                    st.error('\n'.join(response_json['messages']))
