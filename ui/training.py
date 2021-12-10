import os
import pprint as pp
from datetime import datetime
from tempfile import TemporaryDirectory

import streamlit as st
from pyspark.sql import SparkSession

from faas.helper import get_trained
from faas.storage import StoredModel, write_model
from faas.utils.io import load_cached_df_from_st_uploaded
from ui.config import get_config
from ui.visualization.vis_df import preview_df
from ui.visualization.vis_iid import vis_ui_iid
from ui.visualization.vis_lightgbm import get_vis_lgbmwrapper
from ui.visualization.vis_spatial import vis_ui_spatial
from ui.visualization.vis_ts import vis_ui_ts

spark = (
    SparkSession
    .builder
    .appName('ui_training')
    .config('spark.driver.maxResultsSize', '16g')
    .getOrCreate()
)


def run_training(st_container=None):
    if st_container is None:
        st_container = st

    st_container.title('Training')
    st_container.markdown(
        'Upload a dataset with target and feature columns in order to train a model.')
    training_file = st_container.file_uploader('Training data', type=['csv', 'parquet'])

    if training_file is not None:
        df = load_cached_df_from_st_uploaded(f=training_file, spark=spark)

        st_container.header('Uploaded dataset')
        preview_df(df=df, st_container=st_container)

        conf = get_config(df=df, st_container=st_container)

        if conf is not None:

            if st_container.checkbox('Show scatterplot'):
                vis_ui_iid(df=df, config=conf, st_container=st_container)
            if conf.date_column is not None:
                if st_container.checkbox('Show time series plot'):
                    vis_ui_ts(df=df, config=conf, st_container=st_container)
            if conf.has_spatial_columns:
                if st_container.checkbox('Show spatial plot'):
                    vis_ui_spatial(df=df, config=conf, st_container=st_container)

            st_container.header('Current configuration')
            st_container.code(pp.pformat(conf.__dict__, compact=True))

            if st_container.button('Train'):

                with st.spinner('Running training...'):
                    m = get_trained(conf=conf, df=df)
                    stored_model = StoredModel(dt=datetime.now(), m=m, config=conf)

                # write fitted model and st_containerore model key in user session
                model_key = write_model(stored_model)
                st.session_state['model_key'] = model_key

                st_container.success(
                    'Model trained! Model key can be now be used for Prediction.')
                st_container.markdown(f'Model key: `{model_key}`')
                with st_container.expander('Model visualization'):
                    get_vis_lgbmwrapper(m)
