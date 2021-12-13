from typing import Optional

import plotly.express as px
import streamlit as st
from plotly.graph_objs._figure import Figure
from pyspark.sql import DataFrame

from faas.config import Config
from ui.predict import PREDICTION_COLUMN


def plot_evaluate_iid(
    df_evaluation: DataFrame,
    config: Config,
    color_feature: Optional[str] = None
) -> Figure:
    select_cols = [PREDICTION_COLUMN, config.target]
    if color_feature is not None:
        select_cols.append(color_feature)
    pdf = df_evaluation.select(*select_cols).toPandas()

    if config.target_is_categorical:
        fig = px.density_heatmap(pdf, x=config.target, y=PREDICTION_COLUMN)
    else:
        fig = px.scatter(pdf, x=config.target, y=PREDICTION_COLUMN, color=color_feature)
    return fig


def vis_evaluate_iid(df_evaluation: DataFrame, config: Config):
    # color_feature is counts by default for categorical target
    color_feature = None
    if not config.target_is_categorical:
        color_feature = st.selectbox(
            'Color Feature', options=sorted(config.feature_columns))

    st.markdown('''
    Here we plot actuals on the horizontal axis against the predictions on the vertical axis.
    A perfect prediction would show up as a 45-degree line, where every prediction is exactly the
    same as actuals.
    ''')
    st.plotly_chart(plot_evaluate_iid(
        df_evaluation=df_evaluation,
        config=config,
        color_feature=color_feature
    ))
