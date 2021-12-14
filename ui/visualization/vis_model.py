import pprint as pp

import pandas as pd
import plotly.express as px
import streamlit as st
from lightgbm import LGBMModel
from plotly.graph_objs._figure import Figure

from faas.storage import StoredModel
from faas.transformer.lightgbm import ETLWrapperForLGBM


def vis_importance(m: LGBMModel) -> Figure:
    df = pd.DataFrame({
        'name': m.feature_name_,
        'importance': m.feature_importances_,
    })
    lowest_decile = .1 * df['importance'].sum()
    df.loc[df['importance'] <= lowest_decile, 'name'] = 'other features'
    if df['importance'].max() == 0:
        # plot other features
        df['importance'] = 1.
    fig = px.pie(df, values='importance', names='name', title='Feature Importance')
    return fig


def vis_lgbmwrapper(m: ETLWrapperForLGBM):
    st.plotly_chart(vis_importance(m.m))


def vis_stored_model(stored_model: StoredModel):
    st.markdown(f'''
    Created on: `{stored_model.dt}`

    Config:
    ```python
    {pp.pformat(stored_model.config.__dict__, compact=True)}
    ```

    ETLConfig:
    ```python
    {pp.pformat(stored_model.m.config.to_dict(), compact=True)}
    ```
    ''')
    vis_lgbmwrapper(stored_model.m)
