import os
import pprint as pp
from datetime import datetime
from tempfile import TemporaryDirectory

import streamlit as st
from pyspark.sql import SparkSession

from faas.helper import get_trained
from faas.storage import StoredModel, write_model
from faas.utils.io import dump_file_to_location
from faas.utils.types import load_csv, load_parquet
from ui.config import get_config
from ui.visualization.vis_df import preview_df
from ui.visualization.vis_iid import vis_ui_iid
from ui.visualization.vis_lightgbm import get_vis_lgbmwrapper
from ui.visualization.vis_spatial import vis_ui_spatial
from ui.visualization.vis_ts import vis_ui_ts

spark = SparkSession.builder.appName('ui_training').getOrCreate()


def run_training():
    st.title('Training')
    st.markdown('Upload a dataset with target and feature columns in order to train a model.')
    training_file = st.file_uploader('Training data', type=['csv', 'parquet'])

    if training_file is not None:
        with TemporaryDirectory() as temp_dir:

            # load training_file as spark dataframe
            if training_file.name.endswith('.csv'):
                training_path = os.path.join(temp_dir, 'train.csv')
                dump_file_to_location(training_file, p=training_path)
                df = load_csv(spark=spark, p=training_path)
            elif training_file.name.endswith('.parquet'):
                training_path = os.path.join(temp_dir, 'train.parquet')
                dump_file_to_location(training_file, p=training_path)
                df = load_parquet(spark=spark, p=training_path)

            st.header('Uploaded dataset')
            preview_df(df=df)

            conf = get_config(df=df)

            if conf is not None:

                st.header('Exploratory analysis')
                vis_ui_iid(df=df, config=conf)
                if conf.date_column is not None:
                    with st.expander('Time Series Visualization'):
                        vis_ui_ts(df=df, config=conf)
                if conf.has_spatial_columns:
                    with st.expander('Spatial Visualization'):
                        vis_ui_spatial(df=df, config=conf)

                st.header('Current configuration')
                st.code(pp.pformat(conf.__dict__, compact=True))

                if st.button('Train'):
                    m = get_trained(conf=conf, df=df)
                    stored_model = StoredModel(dt=datetime.now(), m=m, config=conf)
                    with st.expander('Model visualization'):
                        get_vis_lgbmwrapper(m)

                    # write fitted model
                    model_key = write_model(stored_model)
                    st.success('Model trained! Model key can be now be used for Prediction.')
                    st.markdown(f'Model key: `{model_key}`')

                    # store model key in user session
                    st.session_state['model_key'] = model_key
