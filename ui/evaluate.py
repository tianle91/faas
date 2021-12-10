import streamlit as st
from pyspark.sql import DataFrame

from faas.storage import StoredModel
from ui.evaluation.utils import validate_evaluation
from ui.evaluation.vis_iid import vis_evaluate_iid
from ui.evaluation.vis_spatial import vis_evaluate_spatial
from ui.evaluation.vis_ts import vis_evaluate_ts
from ui.predict import PREDICTION_COLUMN, highlight_columns
from ui.visualization.vis_lightgbm import get_vis_lgbmwrapper


def run_evaluation(st_container=None):
    if st_container is None:
        st_container = st

    st_container.title('Evaluation')

    df_evaluation: DataFrame = st.session_state.get('df_evaluation', None)
    stored_model: StoredModel = st.session_state.get('stored_model', None)
    if df_evaluation is None:
        st_container.error('Upload dataframe with actuals and run predictions for evaluation.')
        return None
    if stored_model is None:
        st_container.error('No stored model, please provide a model key.')

    get_vis_lgbmwrapper(stored_model.m, st_container=st_container)

    config = stored_model.config
    st_container.markdown(f'Target feature: `{config.target}`')

    pdf_evaluation = (
        df_evaluation
        .select(config.target, PREDICTION_COLUMN, *config.feature_columns)
        .limit(100)
        .toPandas()
    )
    st_container.dataframe(highlight_columns(
        pdf_evaluation,
        columns=[config.target, PREDICTION_COLUMN]
    ))

    validate_evaluation(df=df_evaluation, config=config)

    st_container.header('Scatterplot')
    vis_evaluate_iid(df_evaluation=df_evaluation, config=config, st_container=st_container)
    if config.date_column is not None:
        st_container.header('Time Series')
        vis_evaluate_ts(df_evaluation=df_evaluation, config=config, st_container=st_container)
    if config.has_spatial_columns:
        st_container.header('Spatial')
        vis_evaluate_spatial(df_evaluation=df_evaluation, config=config, st_container=st_container)
