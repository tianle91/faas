import logging
import os
import pprint as pp
from tempfile import TemporaryDirectory

import streamlit as st
from pyspark.sql import SparkSession

from faas.config.config import create_etl_config
from faas.lightgbm import LGBMWrapper
from faas.storage import write_model
from faas.utils.io import dump_file_to_location
from faas.utils.types import load_csv
from ui.config import get_config
from ui.vis_lightgbm import get_vis_lgbmwrapper

logger = logging.getLogger(__name__)


def run_training():

    spark = SparkSession.builder.getOrCreate()

    st.title('Training')
    training_file = st.file_uploader('Training data', type='csv')

    with TemporaryDirectory() as temp_dir:
        if training_file is not None:
            # get the file into a local path
            training_path = os.path.join(temp_dir, 'train.csv')
            dump_file_to_location(training_file, p=training_path)
            df = load_csv(spark=spark, p=training_path)

            conf = get_config(df=df)

            st.header('Current configuration')
            st.code(pp.pprint(conf.__dict__))

            if st.button('Train now!'):
                etl_conf = create_etl_config(conf=conf, df=df)
                m = LGBMWrapper(config=etl_conf)
                m.fit(df)

                with st.expander('Model visualization'):
                    get_vis_lgbmwrapper(m)

                # write fitted model
                model_key = write_model(m)
                st.session_state['model_key'] = model_key
                st.success(f'Model key (save this for prediction): `{model_key}`')
                logger.info(f'wrote model key: {model_key}')
