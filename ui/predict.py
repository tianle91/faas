import os
from tempfile import TemporaryDirectory

import streamlit as st
from pyspark.sql import SparkSession

from faas.e2e import plot_feature_importances
from faas.evaluation.iid import plot_target_scatter
from faas.storage import read_model
from faas.utils.dataframe import JoinableByRowID
from faas.utils.io import dump_file_to_location
from ui.checklist import run_features_checklist, run_target_checklist


def run_predict():

    spark = SparkSession.builder.getOrCreate()

    st.title('Predict')

    e2e = st.session_state.get('trained_model', None)
    key = st.text_input('Model key (obtain this from training)')
    if key is not None and key != '':
        try:
            e2e = read_model(key=key)
            st.session_state['trained_model'] = e2e
        except KeyError:
            st.error(f'Key {key} not found!')

    if e2e is not None:
        with st.expander('Details on loaded model'):
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
            dump_file_to_location(predict_file, p=predict_path)
            df = spark.read.options(header=True, inferSchema=True).csv(predict_path)
            df = JoinableByRowID(df).df

            all_good_x = run_features_checklist(e2e, df=df)
            with st.expander('Details on uploaded dataset'):
                st.write(df.limit(10).toPandas())

            if all_good_x:
                st.markdown('## Predict Now?')
                if st.button('Yes'):
                    df_predict = e2e.predict(df)
                    st.write(df_predict.limit(10).toPandas())
                    st.download_button(
                        'Download prediction',
                        data=df_predict.toPandas().to_csv(),
                        file_name='prediction.csv'
                    )
                    if e2e.target_column in df.columns:
                        with st.expander(f'Found target: {e2e.target_column}. See evaluation?'):
                            all_good_target = run_target_checklist(e2e, df=df)
                            if all_good_target:
                                st.pyplot(
                                    plot_target_scatter(
                                        df_prediction=(
                                            df_predict
                                            .select(e2e.target_column)
                                            .toPandas()
                                        ),
                                        df_actual=(
                                            df
                                            .select(e2e.target_column)
                                            .toPandas()
                                        ),
                                        target_column=e2e.target_column,
                                    )
                                )
