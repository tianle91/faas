import streamlit as st
from pyspark.sql import DataFrame

from faas.storage import read_model
from faas.utils.dataframe import has_duplicates
from ui.evaluation.vis_iid import vis_evaluate_iid


def run_evaluation():
    st.title('Evaluation')

    df_predict: DataFrame = st.session_state.get('df_predict', None)
    df_actual: DataFrame = st.session_state.get('df_actual', None)

    if df_predict is None or df_actual is None:
        st.error('Upload dataframe with actuals and run predictions for evaluation.')
    else:
        model_key = st.session_state.get('model_key')
        stored_model = read_model(key=model_key)
        config = stored_model.config

        config.validate_df(df=df_predict)
        config.validate_df(df=df_actual)

        if has_duplicates(df_predict.select(*config.used_columns_prediction)):
            raise ValueError('Cannot evaluate as df_predict has duplicate feature columns.')
        if has_duplicates(df_actual.select(*config.used_columns_prediction)):
            raise ValueError('Cannot evaluate as df_actual has duplicate feature columns.')

        vis_evaluate_iid(df_predict=df_predict, df_actual=df_actual, config=config)
