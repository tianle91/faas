import os
import pprint as pp
from tempfile import TemporaryDirectory

import streamlit as st
from pyspark.sql import SparkSession

from faas.config.config import create_etl_config
from faas.lightgbm import ETLWrapperForLGBM
from faas.storage import write_model
from faas.utils.io import dump_file_to_location
from faas.utils.types import load_csv
from ui.config import get_config
from ui.vis_lightgbm import get_vis_lgbmwrapper

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

            conf = get_config(df=df)

            if conf is not None:
                st.header('Current configuration')
                st.code(pp.pformat(conf.__dict__, compact=True))

                if st.button('Train'):
                    lgbmw = ETLWrapperForLGBM(config=create_etl_config(conf=conf, df=df))
                    lgbmw.fit(df)

                    with st.expander('Model visualization'):
                        get_vis_lgbmwrapper(lgbmw)

                    # write fitted model
                    model_key = write_model(model=lgbmw, conf=conf)
                    st.success(f'Model key (save this for prediction): `{model_key}`')

                    # store model key in user session
                    st.session_state['model_key'] = model_key
