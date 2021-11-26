import json
import os
import pprint as pp
from tempfile import TemporaryDirectory

import pandas as pd
import requests
import streamlit as st

from api import PredictionResponse
from faas.storage import read_model
from faas.utils.io import dump_file_to_location
from ui.vis_lightgbm import get_vis_lgbmwrapper

APIURL = os.getenv('APIURL', default='http://localhost:8000')


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

                # load the csv
                pred_pdf = pd.read_csv(predict_path)

                st.header('Uploaded dataset')
                preview_n = 100
                st.markdown(f'Preview for first {preview_n} rows out of {len(pred_pdf)} loaded.')
                st.dataframe(pred_pdf.head(preview_n))

                if st.button('Predict'):
                    st.header('Prediction')

                    # send to api
                    r = requests.post(
                        url=f'{APIURL}/predict',
                        data=json.dumps({
                            'model_key': model_key,
                            'data': pred_pdf.to_dict(orient='records')
                        })
                    )
                    prediction_response = PredictionResponse(**r.json())

                    if prediction_response.num_calls_remaining is not None:
                        st.markdown(
                            f'num_calls_remaining: `{prediction_response.num_calls_remaining}`')

                    if prediction_response.prediction is None:
                        st.error('Errors encountered')
                    else:
                        pred_pdf_received = pd.DataFrame(prediction_response.prediction)

                        # preview
                        st.markdown(
                            f'Preview for first {preview_n} rows '
                            f'out of {len(pred_pdf_received)} predictions.'
                        )
                        preview_columns = [
                            stored_model.config.target, *stored_model.config.feature_columns]
                        pred_pdf_preview = pred_pdf_received[preview_columns].head(preview_n)
                        st.dataframe(pred_pdf_preview.style.apply(
                            lambda s: highlight_target(s, target_column=stored_model.config.target),
                            axis=0
                        ))

                        # download
                        st.download_button(
                            f'Download all {len(pred_pdf_received)} predictions.',
                            data=pred_pdf_received.to_csv(),
                            file_name='prediction.csv'
                        )

                    st.markdown('\n\n'.join(
                        [f'❌ {msg}' for msg in prediction_response.messages]))
