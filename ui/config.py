import logging

import streamlit as st
from pyspark.sql import DataFrame
from pyspark.sql.types import NumericType, StringType

from faas.config import Config
from faas.config.utils import get_columns_by_type

logger = logging.getLogger(__name__)


def get_config(df: DataFrame) -> Config:
    st.header('Create a configuration')
    numeric_columns = get_columns_by_type(df=df, dtype=NumericType)
    categorical_columns = get_columns_by_type(df=df, dtype=StringType)
    logger.info(f'numeric_columns: {numeric_columns}')
    logger.info(f'categorical_columns: {categorical_columns}')

    target_column = st.selectbox('target column', options=numeric_columns + categorical_columns)
    target_is_categorical = target_column in categorical_columns

    with st.expander('Is there a date column?'):
        date_column = st.selectbox(
            'Date column',
            options=[None, *categorical_columns],
        )
        date_column_format = st.selectbox(
            'Date format',
            options=['yyyy-MM-dd'],
            help='https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.sql.functions.to_date.html'
        )

    with st.expander('Are there spatial columns?'):
        latitude_column = st.selectbox(
            'Latitude Column (-90. to 90.)',
            options=[None, *numeric_columns],
        )
        longitude_column = st.selectbox(
            'Longitude Column (-180. to 180.)',
            options=[None, *numeric_columns],
        )

    group_columns = None
    with st.expander('Is there a group column?'):
        group_columns = st.multiselect(
            'Group Columns (only categorical columns can be used as groups)',
            options=[c for c in categorical_columns if c != date_column],
            default=None
        )

    used_columns = [target_column, date_column, latitude_column, longitude_column]
    if group_columns is not None:
        used_columns += group_columns
    possible_feature_columns = [
        c for c in numeric_columns + categorical_columns
        if c not in used_columns
    ]
    feature_columns = st.multiselect(
        label='Feature Columns',
        options=possible_feature_columns,
        default=possible_feature_columns,
        help='Including all available features is helpful for model training.'
    )

    conf = Config(
        target=target_column,
        target_is_categorical=target_is_categorical,
        date_column=date_column,
        date_column_format=date_column_format,
        latitude_column=latitude_column,
        longitude_column=longitude_column,
        group_columns=group_columns,
        feature_columns=feature_columns
    )
    return conf
