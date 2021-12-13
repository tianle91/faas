import plotly.express as px
import pyspark.sql.functions as F
import streamlit as st
from plotly.graph_objs._figure import Figure
from pyspark.sql import DataFrame

from faas.config import Config
from ui.predict import PREDICTION_COLUMN

IS_PREDICTION_COLUMN = '__IS_PREDICTION__'


def plot_ts(df_evaluation: DataFrame, config: Config) -> Figure:
    actual_df = (
        df_evaluation
        .withColumn(IS_PREDICTION_COLUMN, F.lit(False))
        .select(config.date_column, config.target, IS_PREDICTION_COLUMN)
    )
    pred_df = (
        df_evaluation
        .withColumn(config.target, F.col(PREDICTION_COLUMN))
        .withColumn(IS_PREDICTION_COLUMN, F.lit(True))
        .select(config.date_column, config.target, IS_PREDICTION_COLUMN)
    )

    # what do we need?
    pdf = actual_df.union(pred_df).toPandas()
    fig = px.scatter(
        pdf,
        x=config.date_column,
        y=config.target,
        marginal_y='histogram',
        color=IS_PREDICTION_COLUMN
    )
    return fig


def vis_evaluate_ts(df_evaluation: DataFrame, config: Config):
    st.markdown('''
    We plot predicted values against actual values.
    A good prediction would show up as the predicted lines being close to actual lines.
    ''')
    st.plotly_chart(plot_ts(
        df_evaluation=df_evaluation,
        config=config,
    ))
