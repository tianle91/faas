import os
import pickle
from tempfile import TemporaryDirectory

import pandas as pd
import streamlit as st
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import NumericType

from faas.e2e import E2EPipline, plot_feature_importances
from faas.loss import plot_prediction_vs_actual
from faas.utils_dataframe import JoinableByRowID


def run_checklist(e2e: E2EPipline, df: DataFrame) -> bool:

    target_is_numeric = isinstance(df.schema[e2e.target_column].dataType, NumericType)
    target_column_passed = (e2e.target_is_numeric == target_is_numeric)

    feature_columns_passed = all([c in df.columns for c in e2e.feature_columns])

    numeric_features_passed = [
        isinstance(df.schema[c].dataType, NumericType)
        for c in e2e.numeric_features
    ]

    categorical_features_passed = [
        not isinstance(df.schema[c].dataType, NumericType)
        for c in e2e.categorical_features
    ]

    all_good = all([
        target_column_passed,
        feature_columns_passed,
        numeric_features_passed,
        categorical_features_passed
    ])
    st.markdown('Target column: ' + '✅' if target_column_passed else '❌')
    st.markdown('Feature columns: ' + '✅' if feature_columns_passed else '❌')
    st.markdown('Numeric columns: ' + '✅' if numeric_features_passed else '❌')
    st.markdown('Categorical columns: ' + '✅' if categorical_features_passed else '❌')
    st.markdown('All good: ' + '✅' if all_good else '❌')
    return all_good


def run_predict():

    spark = SparkSession.builder.getOrCreate()

    st.title('Predict')

    e2e = st.session_state.get('trained_model', None)
    upload_trained = st.file_uploader('Upload trained', type='model')
    if upload_trained is not None:
        e2e: E2EPipline = pickle.loads(upload_trained.getvalue())
        st.session_state['trained_model'] = e2e

    if e2e is not None:
        st.pyplot(plot_feature_importances(m=e2e.m))
        st.markdown(f'Target column: `{e2e.target_column}`')
        formatted_feature_cols = ' '.join([f'`{c}`' for c in e2e.feature_columns])
        st.markdown(f'Feature columns: {formatted_feature_cols}')

    st.markdown('# Upload dataset')
    predict_file = st.file_uploader('Predict data', type='csv')

    with TemporaryDirectory() as temp_dir:
        if predict_file is not None:
            # get the file into a local path
            predict_path = os.path.join(temp_dir, 'predict.csv')
            pd.read_csv(predict_file).to_csv(predict_path, index=False)

            df = spark.read.options(header=True, inferSchema=True).csv(predict_path)
            df = JoinableByRowID(df).df

            st.markdown('# Uploaded dataset')
            st.write(df.limit(10).toPandas())

            st.markdown('## Checklist')
            all_good = run_checklist(e2e, df=df)

            if all_good:
                st.markdown('# Predict Now?')
                if st.button('Yes'):
                    df_predict = e2e.predict(df)
                    st.write(df_predict.limit(10).toPandas())
                    st.download_button(
                        'Download prediction',
                        data=df_predict.toPandas().to_csv(),
                        file_name='prediction.csv'
                    )
                    if e2e.target_column in df.columns:
                        st.pyplot(
                            plot_prediction_vs_actual(
                                df_prediction=df_predict.select(e2e.target_column).toPandas(),
                                df_actual=df.select(e2e.target_column).toPandas(),
                                column=e2e.target_column,
                            )
                        )
