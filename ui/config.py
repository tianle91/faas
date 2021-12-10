import logging

import streamlit as st
from pyspark.sql import DataFrame
from pyspark.sql.types import DateType, NumericType, StringType, TimestampType

from faas.config import Config
from faas.config.utils import get_columns_by_type

logger = logging.getLogger(__name__)


def get_config(df: DataFrame) -> Config:
    numeric_columns = get_columns_by_type(df=df, dtype=NumericType)
    categorical_columns = get_columns_by_type(df=df, dtype=StringType)
    timestamp_columns = get_columns_by_type(df=df, dtype=TimestampType)
    timestamp_columns += get_columns_by_type(df=df, dtype=DateType)

    summary_pdf = df.describe().toPandas().set_index(keys='summary')
    degenerate_columns = [
        c for c in df.columns
        if c in summary_pdf.columns and int(summary_pdf.loc['count', c]) <= 1
    ]
    if len(degenerate_columns) > 0:
        st.warning(
            f'Degenerate columns: {degenerate_columns} '
            'cannot be used in target and feature columns'
        )

    st.header("What's the target column?")
    target_column = st.selectbox(
        'target column',
        options=sorted([
            c for c in numeric_columns + categorical_columns
            if c not in degenerate_columns
        ])
    )
    target_is_categorical = target_column in categorical_columns

    st.header('Any special columns?')
    with st.expander('Date columns enable date-related features and visualization'):
        date_column = st.selectbox(
            'Date column (yyyy-MM-dd)',
            options=[None, ] + sorted(categorical_columns + timestamp_columns),
        )
    with st.expander('Spatial columns enable location features and map visualization'):
        latitude_column = st.selectbox(
            'Latitude Column (-90. to 90.)',
            options=[None, ] + sorted(numeric_columns),
        )
        longitude_column = st.selectbox(
            'Longitude Column (-180. to 180.)',
            options=[None, ] + sorted(numeric_columns),
        )
    with st.expander('Groups allow training and visualization to be focused on important segments of data'):
        group_columns = st.multiselect(
            'Group Columns (only categorical columns can be used as groups)',
            options=sorted([
                c for c in categorical_columns
                if c not in [date_column, target_column]
            ]),
            default=[]
        )
        num_groups = df.select(group_columns).distinct().count()
        st.info(f'There are {num_groups} groups')
    if len(group_columns) == 0:
        group_columns = None

    st.header('Feature columns')
    used_columns = [target_column, date_column, latitude_column, longitude_column]
    possible_feature_columns = sorted([
        c for c in numeric_columns + categorical_columns
        if c not in used_columns + degenerate_columns
    ])
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
    st.markdown(f'Number of features: {len(feature_columns)}')

    # group columns must be used as features
    actual_feature_columns = feature_columns
    if group_columns is not None:
        actual_feature_columns += group_columns
    actual_feature_columns = [
        c for c in set(actual_feature_columns)
        if c not in degenerate_columns
    ]

    conf = Config(
        target=target_column,
        target_is_categorical=target_is_categorical,
        date_column=date_column,
        latitude_column=latitude_column,
        longitude_column=longitude_column,
        group_columns=group_columns,
        feature_columns=actual_feature_columns
    )
    return conf
