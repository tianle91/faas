import logging
import os
from tempfile import TemporaryDirectory

import streamlit as st
from pyspark.sql import SparkSession

from faas.config import ETLConfig, recommend
from faas.lightgbm import ETLWrapperForLGBM
from faas.storage import write_model
from faas.utils.io import dump_file_to_location
from faas.utils.types import DEFAULT_DATE_FORMAT, load_csv

logger = logging.getLogger(__name__)


def run_training():

    spark = SparkSession.builder.getOrCreate()

    st.title('Training')
    st.markdown(f'Ensure that dates are in the `{DEFAULT_DATE_FORMAT}` format.')
    training_file = st.file_uploader('Training data', type='csv')

    with TemporaryDirectory() as temp_dir:
        if training_file is not None:
            # get the file into a local path
            training_path = os.path.join(temp_dir, 'train.csv')
            dump_file_to_location(training_file, p=training_path)
            df = load_csv(spark=spark, p=training_path)

            st.header('What to train?')
            st.dataframe(df.limit(10).toPandas())

            target_column = st.selectbox('target column', options=df.columns)
            if st.button('Get configuration'):
                config, messages = recommend(df=df, target_column=target_column)
                st.markdown('\n'.join([f'- {message}' for message in messages]))
                st.session_state['config'] = config

            st.header('Current configuration')
            config = st.session_state.get('config', None)
            if config is not None:
                config: ETLConfig = config
                st.write(config.get_markdown())
                if st.button('Train now!'):
                    e = ETLWrapperForLGBM(config=config).fit(df)
                    st.session_state['model'] = e
                    key = write_model(e)
                    st.success(f'Model key (save this for prediction): `{key}`')
                    logger.info(f'wrote model key: {key}')
