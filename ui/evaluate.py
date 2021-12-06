import streamlit as st
from pyspark.sql import DataFrame

from faas.storage import read_model
from ui.evaluation.utils import validate_evaluation
from ui.evaluation.vis_iid import vis_evaluate_iid
from ui.evaluation.vis_ts import vis_evaluate_ts


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

        validate_evaluation(df=df_predict, config=config)
        validate_evaluation(df=df_actual, config=config)

        vis_evaluate_iid(df_predict=df_predict, df_actual=df_actual, config=config)

        if config.date_column is not None:
            with st.expander('Time Series Visualization'):
                vis_evaluate_ts(df_predict=df_predict, df_actual=df_actual, config=config)
