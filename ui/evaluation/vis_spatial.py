import plotly.express as px
import pyspark.sql.functions as F
import streamlit as st
from plotly.graph_objs._figure import Figure
from pyspark.sql import DataFrame

from faas.config import Config
from ui.predict import PREDICTION_COLUMN

ERROR_COL = '__ERROR__'


def categorical_error(
    df: DataFrame, actual_col: str, predict_col: str, error_col: str = ERROR_COL
) -> DataFrame:
    return df.withColumn(
        error_col,
        F.when(F.col(actual_col) == F.col(predict_col), F.lit(0)).otherwise(F.lit(1))
    )


def numeric_error(
    df: DataFrame, actual_col: str, predict_col: str, error_col: str = ERROR_COL
) -> DataFrame:
    return df.withColumn(
        error_col,
        (F.col(predict_col) - F.col(actual_col)) / F.col(actual_col)
    )


def plot_spatial(df_evaluation: DataFrame, config: Config) -> Figure:

    if config.target_is_categorical:
        f = categorical_error
    else:
        f = numeric_error

    df_evaluation = f(
        df=df_evaluation,
        actual_col=config.target,
        predict_col=PREDICTION_COLUMN,
        error_col=ERROR_COL
    )

    select_cols = config.used_columns_prediction + [config.target, PREDICTION_COLUMN, ERROR_COL]

    pdf = df_evaluation.select(*select_cols).toPandas()
    fig = px.scatter_geo(
        pdf,
        lat=config.latitude_column,
        lon=config.longitude_column,
        color=ERROR_COL,
        hover_data=[config.target, PREDICTION_COLUMN, ERROR_COL],
        fitbounds='locations'
    )
    return fig


def vis_evaluate_spatial(df_evaluation: DataFrame, config: Config):
    st.markdown('''
    High error values indicate that predictions are far from actuals.
    ''')
    st.plotly_chart(plot_spatial(
        df_evaluation=df_evaluation,
        config=config,
    ))
