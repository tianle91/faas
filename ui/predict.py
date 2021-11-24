import os
import pprint as pp
from tempfile import TemporaryDirectory
from typing import Optional

import pandas as pd
import pyspark.sql.functions as F
import streamlit as st
from pyspark.sql import SparkSession

from faas.lightgbm import LGBMWrapper
from faas.storage import read_model
from faas.utils.dataframe import JoinableByRowID
from faas.utils.io import dump_file_to_location


def highlight_target(s: pd.Series, target_column: str):
    if s.name == target_column:
        return ['background-color: #ff0000'] * len(s)
    else:
        return ['background-color: #000000'] * len(s)


def run_predict():

    spark = SparkSession.builder.getOrCreate()

    st.title('Predict')
    m: Optional[LGBMWrapper] = st.session_state.get('model', None)
    if m is None:
        key = st.text_input('Model key (obtain this from training)')
        if key != '':
            try:
                m: LGBMWrapper = read_model(key=key)
                st.session_state['model'] = m
            except KeyError:
                st.error(f'Key {key} not found!')
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

                df = spark.read.options(header=True, inferSchema=True).csv(predict_path)

                # date
                date_column = m.config.weight.date_column
                if date_column is not None:
                    df = df.withColumn(date_column, F.to_date(date_column))

                df = JoinableByRowID(df).df

                ok, msgs = m.check_df_prediction(df=df)
                if ok:
                    st.success('Uploaded dataset is valid!')
                else:
                    st.error('\n'.join(msgs))

                if ok:
                    st.markdown('## Predict Now?')
                    if st.button('Yes'):
                        df_predict = m.predict(df)
                        df_predict_preview: pd.DataFrame = (
                            df_predict
                            .select(
                                *m.config.feature.categorical_columns,
                                *m.config.feature.numeric_columns,
                                m.config.target.column
                            )
                            .limit(10)
                            .toPandas()
                        )
                        st.dataframe(df_predict_preview.style.apply(
                            lambda s: highlight_target(s, target_column=m.config.target.column),
                            axis=0
                        ))
                        st.download_button(
                            'Download prediction',
                            data=df_predict.toPandas().to_csv(),
                            file_name='prediction.csv'
                        )
