import os
import pprint as pp
from datetime import datetime
from tempfile import TemporaryDirectory

import streamlit as st
from pyspark.sql import SparkSession

from faas.helper import get_trained
from faas.storage import StoredModel, write_model
from faas.utils.io import dump_file_to_location
from faas.utils.types import load_csv
from ui.config import get_config
from ui.visualization.vis_df import preview_df
from ui.visualization.vis_iid import vis_ui_iid
from ui.visualization.vis_lightgbm import get_vis_lgbmwrapper
from ui.visualization.vis_ts import vis_ui_ts

spark = SparkSession.builder.appName('ui_training').getOrCreate()


def run_training():
    st.title('Training')
    training_file = st.file_uploader('Training data', type='csv')

    with TemporaryDirectory() as temp_dir:
        if training_file is not None:
            # get the file into a local path
            training_path = os.path.join(temp_dir, 'train.csv')
            dump_file_to_location(training_file, p=training_path)

            df = load_csv(spark=spark, p=training_path)

            st.header('Uploaded dataset')
            preview_df(df=df)

            conf = get_config(df=df)

            if conf is not None:
                with st.expander('Visualization'):
                    if conf.date_column is not None:
                        vis_ui_ts(df=df, config=conf)
                    if not conf.has_spatial_columns:
                        vis_ui_iid(df=df, config=conf)

                st.header('Current configuration')
                st.code(pp.pformat(conf.__dict__, compact=True))

                if st.button('Train'):
                    m = get_trained(conf=conf, df=df)
                    stored_model = StoredModel(dt=datetime.now(), m=m, config=conf)
                    with st.expander('Model visualization'):
                        get_vis_lgbmwrapper(m)

                    # write fitted model
                    model_key = write_model(stored_model)
                    st.success(f'Model key (save this for prediction): `{model_key}`')

                    # store model key in user session
                    st.session_state['model_key'] = model_key
