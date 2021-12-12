import logging

import pyspark.sql.functions as F
import streamlit as st
from pyspark.sql import DataFrame, SparkSession

from faas.config import Config
from faas.helper import get_prediction
from faas.storage import StoredModel, decrement_num_calls_remaining
from faas.utils.dataframe import has_duplicates
from ui.visualization.vis_df import highlight_columns
from ui.visualization.vis_lightgbm import get_vis_lgbmwrapper

logger = logging.getLogger(__name__)

spark = (
    SparkSession
    .builder
    .appName('ui_predict')
    .config('spark.driver.maxResultsSize', '16g')
    .getOrCreate()
)

PREDICTION_COLUMN = '__PREDICTION__'


def preview_prediction(df_predict: DataFrame, config: Config, n: int = 100, st_container=None):
    if st_container is None:
        st_container = st

    # nicely order the preview columns
    preview_columns = [config.target]
    if config.date_column is not None:
        preview_columns.append(config.date_column)
    if config.has_spatial_columns:
        preview_columns += [config.latitude_column, config.longitude_column]
    if config.group_columns is not None:
        preview_columns += config.group_columns
    for c in config.feature_columns:
        if c not in preview_columns:
            preview_columns.append(c)

    pdf_predict = df_predict.limit(n).toPandas()
    st_container.markdown(
        f'Preview for first {n}/{df_predict.count()} predictions '
        f'for {config.target}.'
    )
    st_container.dataframe(highlight_columns(pdf_predict[preview_columns], [config.target]))


def run_predict(st_container=None):
    if st_container is None:
        st_container = st

    st_container.title('Predict')

    df: DataFrame = st.session_state.get('df', None)
    stored_model: StoredModel = st.session_state.get('stored_model', None)
    if df is None or stored_model is None:
        if df is None:
            st_container.error('Cannot predict without a dataframe. Please upload one.')
        if stored_model is None:
            st_container.error('Cannot predict without model. Please provide model key.')
        return None

    get_vis_lgbmwrapper(stored_model.m, st_container=st_container)

    if stored_model.num_calls_remaining <= 0:
        st_container.error(f'Num calls remaining: {stored_model.num_calls_remaining}')
        return None

    config = stored_model.config
    st_container.header('Prediction')
    if not st_container.checkbox('Yes', key='prediction_yes'):
        return None

    with st.spinner('Running predictions...'):
        df_predict, msgs = get_prediction(conf=config, df=df, m=stored_model.m)

    st_container.markdown('\n\n'.join([f'❌ {msg}' for msg in msgs]))

    if df_predict is None:
        st_container.error('Unable to generate forecasts')
        return None

    # update num_calls_remaining
    stored_model = decrement_num_calls_remaining(key=st.session_state['model_key'])
    st.session_state['stored_model'] = stored_model
    st_container.info(f'Num calls remaining: {stored_model.num_calls_remaining}')

    # evaluation
    if config.target in df.columns:
        st_container.success(
            f'Detected target column: {config.target} in uploaded dataframe.')
        if has_duplicates(df.select(config.used_columns_prediction)):
            st_container.error(
                'Cannot perform comparison as df has duplicates in '
                f'used_columns_prediction: {config.used_columns_prediction}.'
            )
        else:
            df_predict_renamed = (
                df_predict
                .withColumn(PREDICTION_COLUMN, F.col(config.target))
                .drop(config.target)
            )
            df_evaluation = df.join(
                df_predict_renamed,
                on=config.used_columns_prediction,
                how='left'
            ).cache()
            df_evaluation.count()
            st.session_state['df_evaluation'] = df_evaluation
            st_container.success('Evaluation possible!')

    preview_prediction(df_predict=df_predict, config=config, st_container=st_container)

    if st_container.button('Export predictions'):
        with st_container.spinner('Exporting predictions...'):
            pdf_predict = df_predict.toPandas()
        st_container.download_button(
            f'Download all {df_predict.count()} predictions',
            data=pdf_predict.to_csv(),
            file_name='prediction.csv'
        )
