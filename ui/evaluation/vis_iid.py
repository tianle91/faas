import pandas as pd
import plotly.express as px
import streamlit as st
from plotly.graph_objs._figure import Figure
from pyspark.sql import DataFrame

from faas.config import Config
from ui.predict import PREDICTION_COLUMN


def plot_evaluate_iid(df_evaluation: DataFrame, config: Config) -> Figure:
    select_cols = [PREDICTION_COLUMN, config.target]
    pdf = df_evaluation.select(*select_cols).toPandas()

    if config.target_is_categorical:
        fig = px.density_heatmap(pdf, x=config.target, y=PREDICTION_COLUMN)
    else:
        # this is similar to ui/evaluation/vis_ts.py, consider refactoring
        pdf_actual = pdf.copy()
        pdf_actual['IS_PREDICTION'] = False
        pdf_actual[PREDICTION_COLUMN] = pdf_actual[config.target]
        pdf_actual = pdf_actual[[config.target, PREDICTION_COLUMN, 'IS_PREDICTION']]

        pdf_prediction = pdf.copy()
        pdf_prediction['IS_PREDICTION'] = True
        pdf_prediction = pdf_prediction[[config.target, PREDICTION_COLUMN, 'IS_PREDICTION']]

        pdf_merged = pd.concat([pdf_actual, pdf_prediction], axis=0)

        fig = px.scatter(pdf_merged, x=config.target, y=PREDICTION_COLUMN, color='IS_PREDICTION')
    return fig


def vis_evaluate_iid(df_evaluation: DataFrame, config: Config):
    st.markdown('''
    Here we plot actuals on the horizontal axis against the predictions on the vertical axis.
    A perfect prediction would show up as a 45-degree line, where every prediction is exactly the
    same as actuals (i.e. where `IS_PREDICTION == False`).
    A large gap between predictions and actuals indicates poor model performance.
    ''')
    st.plotly_chart(plot_evaluate_iid(df_evaluation=df_evaluation, config=config))
