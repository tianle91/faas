import streamlit as st
from pyspark.sql import DataFrame
from pyspark.sql.types import NumericType, StringType

from faas.config import Config
from faas.config.utils import get_columns_by_type


def get_config(df: DataFrame) -> Config:
    st.header('Create a configuration')
    st.dataframe(df.limit(10).toPandas())

    numeric_columns = get_columns_by_type(df=df, dtype=NumericType)
    categorical_columns = get_columns_by_type(df=df, dtype=StringType)

    target_column = st.selectbox('target column', options=numeric_columns)

    with st.expander('Is there a date column?'):
        date_column = st.selectbox(
            f'Date column',
            options=[None, *categorical_columns],
        )
        date_column_format = st.selectbox(
            'Date format',
            options=['yyyy-MM-dd'],
            help='https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.sql.functions.to_date.html'
        )

    group_columns = None
    if date_column is not None:
        with st.expander('Is there a group column?'):
            group_columns = st.multiselect(
                'group columns',
                options=[c for c in categorical_columns if c != date_column],
                default=None
            )

    non_target_non_date_columns = [
        c for c in numeric_columns + categorical_columns
        if c not in [target_column, date_column]
    ]
    feature_columns = st.multiselect(
        label='Feature columns',
        options=non_target_non_date_columns,
        default=non_target_non_date_columns,
        help='Including all available features is helpful for model training.'
    )

    conf = Config(
        target=target_column,
        target_is_categorical=~isinstance(df.schema[target_column].dataType, NumericType),
        date_column=date_column,
        date_column_format=date_column_format,
        group_columns=group_columns,
        feature_columns=feature_columns
    )
    return conf
