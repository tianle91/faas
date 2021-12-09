import logging

import streamlit as st
from pyspark.sql import DataFrame
from pyspark.sql.types import NumericType, StringType

from faas.config import Config
from faas.config.utils import get_columns_by_type

logger = logging.getLogger(__name__)


def get_config(df: DataFrame) -> Config:
    numeric_columns = get_columns_by_type(df=df, dtype=NumericType)
    categorical_columns = get_columns_by_type(df=df, dtype=StringType)

    st.header("What's the target column?")
    target_column = st.selectbox('target column', options=numeric_columns + categorical_columns)
    target_is_categorical = target_column in categorical_columns

    st.header('Any special columns?')
    with st.expander('Date columns enable date-related features and visualization'):
        date_column = st.selectbox(
            'Date column',
            options=[None, *categorical_columns],
        )
        date_column_format = st.selectbox(
            'Date format',
            options=['yyyy-MM-dd'],
            help='https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.sql.functions.to_date.html'
        )
    with st.expander('Spatial columns enable location features and map visualization'):
        latitude_column = st.selectbox(
            'Latitude Column (-90. to 90.)',
            options=[None, *numeric_columns],
        )
        longitude_column = st.selectbox(
            'Longitude Column (-180. to 180.)',
            options=[None, *numeric_columns],
        )
    with st.expander('Groups allow training and visualization to be focused on important segments of data'):
        group_columns = st.multiselect(
            'Group Columns (only categorical columns can be used as groups)',
            options=[c for c in categorical_columns if c != date_column],
            default=[]
        )
    if len(group_columns) == 0:
        group_columns = None

    st.header('Feature columns')
    used_columns = [target_column, date_column, latitude_column, longitude_column]
    if group_columns is not None:
        used_columns += group_columns
    possible_feature_columns = [
        c for c in numeric_columns + categorical_columns
        if c not in used_columns
    ]
    st.warning(
        'These columns will all be required at prediction time. '
        'Do not include unavailable ones in training.'
    )
    feature_columns = st.multiselect(
        label='Feature Columns',
        options=possible_feature_columns,
        default=possible_feature_columns,
        help='Including all available features is helpful for model training.'
    )

    # group columns must be used as features
    actual_feature_columns = feature_columns
    if group_columns is not None:
        actual_feature_columns += group_columns
    actual_feature_columns = list(set(actual_feature_columns))

    conf = Config(
        target=target_column,
        target_is_categorical=target_is_categorical,
        date_column=date_column,
        date_column_format=date_column_format,
        latitude_column=latitude_column,
        longitude_column=longitude_column,
        group_columns=group_columns,
        feature_columns=actual_feature_columns
    )
    return conf
