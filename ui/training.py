import logging
import os
import pprint as pp
from tempfile import TemporaryDirectory

import pyspark.sql.functions as F
import streamlit as st
from pyspark.sql import SparkSession
from pyspark.sql.types import NumericType, StringType

from faas.config import Config, get_columns_by_type, recommend_config
from faas.lightgbm import LGBMWrapper
from faas.storage import write_model
from faas.utils.io import dump_file_to_location
from faas.utils.types import DEFAULT_DATE_FORMAT, load_csv

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

            st.header('What to train?')
            st.dataframe(df.limit(10).toPandas())

            numeric_columns = get_columns_by_type(df=df, dtype=NumericType)
            target_column = st.selectbox('target column', options=numeric_columns)

            categorical_columns = get_columns_by_type(df=df, dtype=StringType)

            # date
            st.markdown(f'Ensure that dates are in the `{DEFAULT_DATE_FORMAT}` format.')
            date_column = st.selectbox('date column', options=[None, *categorical_columns])

            # groups
            group_columns = None
            if date_column is not None:
                df = df.withColumn(date_column, F.to_date(date_column))
                non_date_categorical_columns = [c for c in categorical_columns if c != date_column]
                group_columns = st.multiselect(
                    'group columns', options=non_date_categorical_columns, default=None)

            # TODO lon lat columns

            if st.button('Get configuration'):
                config = recommend_config(
                    df=df,
                    target_column=target_column,
                    date_column=date_column,
                    group_columns=group_columns,
                )
                st.session_state['config'] = config

            st.header('Current configuration')
            config = st.session_state.get('config', None)
            if config is not None:
                config: Config = config
                st.code(pp.pformat(config.to_dict()))
                if st.button('Train now!'):
                    m = LGBMWrapper(config=config).fit(df)
                    model_key = write_model(m)
                    st.session_state['model_key'] = model_key
                    st.success(f'Model key (save this for prediction): `{model_key}`')
                    logger.info(f'wrote model key: {model_key}')
