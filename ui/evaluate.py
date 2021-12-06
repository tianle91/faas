import pprint as pp

import streamlit as st
from pyspark.sql import DataFrame

from faas.storage import read_model
from faas.utils.dataframe import has_duplicates
from ui.visualization.vis_iid import vis_evaluate_iid
from ui.visualization.vis_lightgbm import get_vis_lgbmwrapper


def run_evaluation():
    st.title('Evaluation')

    model_key = st.session_state.get('model_key', '')

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

        config = stored_model.config

        df_predict: DataFrame = st.session_state.get('df_predict', None)
        df_actual: DataFrame = st.session_state.get('df_actual', None)

        config.validate_df(df=df_predict)
        config.validate_df(df=df_actual)

        df_predict = df_predict.select(*config.used_columns_prediction)
        if has_duplicates(df_predict):
            raise ValueError('Cannot evaluate as df_predict has duplicate feature columns.')
        df_actual = df_actual.select(*config.used_columns_prediction)
        if has_duplicates(df_actual):
            raise ValueError('Cannot evaluate as df_actual has duplicate feature columns.')

        vis_evaluate_iid(df_predict=df_predict, df_actual=df_actual, config=config)
