import streamlit as st
from pyspark.sql import DataFrame

from faas.storage import StoredModel
from ui.evaluation.utils import validate_evaluation
from ui.evaluation.eval import run_eval
from ui.predict import PREDICTION_COLUMN
from ui.visualization.vis_df import highlight_columns
from ui.visualization.vis_model import vis_stored_model


def run_evaluation():
    st.title('Evaluation')

    df_evaluation: DataFrame = st.session_state.get('df_evaluation', None)
    stored_model: StoredModel = st.session_state.get('stored_model', None)
    if df_evaluation is None or stored_model is None:
        if df_evaluation is None:
            st.error('Run a prediction with actuals for evaluation.')
        if stored_model is None:
            st.error('No stored model, please provide a model key.')
        return None

    config = stored_model.config
    validate_evaluation(df=df_evaluation, config=config)

    with st.expander('Loaded model'):
        vis_stored_model(stored_model=stored_model)

    st.header('Predictions')
    pdf_evaluation = (
        df_evaluation
        .select(config.target, PREDICTION_COLUMN, *config.feature_columns)
        .limit(100)
        .toPandas()
    )
    st.dataframe(highlight_columns(
        pdf_evaluation,
        columns=[config.target, PREDICTION_COLUMN]
    ))

    run_eval(conf=config, df=df_evaluation)
