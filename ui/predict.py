import os
from tempfile import TemporaryDirectory
from typing import Optional

import pandas as pd
import streamlit as st
from pyspark.sql import SparkSession

from faas.e2e.lightgbm import ETLWrapperForLGBM
from faas.storage import read_model
from faas.utils.dataframe import JoinableByRowID
from faas.utils.io import dump_file_to_location
from faas.utils.types import DEFAULT_DATE_FORMAT


def highlight_target(s: pd.Series, target_column: str):
    if s.name == target_column:
        return ['background-color: #ff0000'] * len(s)
    else:
        return ['background-color: #000000'] * len(s)


def run_predict():

    spark = SparkSession.builder.getOrCreate()

    st.title('Predict')

    e: Optional[ETLWrapperForLGBM] = st.session_state.get('model', None)
    key = st.text_input('Model key (obtain this from training)')
    if key != '':
        try:
            e: ETLWrapperForLGBM = read_model(key=key)
            st.session_state['model'] = e
        except KeyError:
            st.error(f'Key {key} not found!')

    if e is not None:
        st.success('Model loaded!')
        with st.expander('Details on loaded model'):
            st.markdown(e.config.get_markdown())

    st.markdown('# Upload dataset')
    st.markdown(f'Ensure that dates are in the `{DEFAULT_DATE_FORMAT}` format.')
    predict_file = st.file_uploader('Predict data', type='csv')

    with TemporaryDirectory() as temp_dir:
        if predict_file is not None:
            # get the file into a local path
            predict_path = os.path.join(temp_dir, 'predict.csv')
            dump_file_to_location(predict_file, p=predict_path)
            df = spark.read.options(header=True, inferSchema=True).csv(predict_path)
            df = JoinableByRowID(df).df

            ok, msgs = e.check_df_prediction(df=df)
            if ok:
                st.success('Uploaded dataset is valid!')
            else:
                st.error('\n'.join(msgs))

            if ok:
                st.markdown('## Predict Now?')
                if st.button('Yes'):
                    df_predict = e.predict(df)
                    st.dataframe((
                        df_predict
                        .select(
                            *e.config.x_categorical_columns,
                            *e.config.x_numeric_features,
                            e.config.target_column
                        )
                        .limit(10)
                        .toPandas()
                        .style
                        .apply(
                            lambda s: highlight_target(s, target_column=e.config.target_column),
                            axis=0
                        )
                    ))
                    st.download_button(
                        'Download prediction',
                        data=df_predict.toPandas().to_csv(),
                        file_name='prediction.csv'
                    )
