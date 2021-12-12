import pprint as pp
from datetime import datetime

import streamlit as st

from faas.helper import get_trained
from faas.storage import StoredModel, write_model
from ui.config import get_config
from ui.visualization.vis_df import preview_df
from ui.visualization.vis_iid import vis_ui_iid
from ui.visualization.vis_model import vis_stored_model
from ui.visualization.vis_spatial import vis_ui_spatial
from ui.visualization.vis_ts import vis_ui_ts


def run_training(st_container=None):
    if st_container is None:
        st_container = st

    st_container.title('Training')
    st_container.markdown(
        'Upload a dataset with target and feature columns in order to train a model.')

    df = st.session_state.get('df', None)

    if df is not None:
        st_container.header('Uploaded dataset')
        preview_df(df=df, st_container=st_container)

        st_container.header('Train a model')
        if not st_container.checkbox('Yes', key='training_yes'):
            return None

        conf = get_config(df=df, st_container=st_container)

        st_container.header('Exploratory analysis')
        if st_container.checkbox('Show scatterplot'):
            vis_ui_iid(df=df, config=conf, st_container=st_container)
        if conf.date_column is not None:
            if st_container.checkbox('Show time series plot'):
                vis_ui_ts(df=df, config=conf, st_container=st_container)
        if conf.has_spatial_columns:
            if st_container.checkbox('Show spatial plot'):
                vis_ui_spatial(df=df, config=conf, st_container=st_container)

        st_container.header('Current configuration')
        st_container.code(pp.pformat(conf.__dict__, compact=True))

        if st_container.button('Train'):

            with st.spinner('Running training...'):
                m = get_trained(conf=conf, df=df)
                stored_model = StoredModel(dt=datetime.now(), m=m, config=conf)

            # write fitted model and store model key in user session
            model_key = write_model(stored_model)
            st.session_state['model_key'] = model_key
            st.session_state['stored_model'] = stored_model

            st_container.success(
                'Model trained! Model key can be now be used for Prediction.')
            st_container.markdown(f'Model key: `{model_key}`')

            st_container.header('Trained model')
            vis_stored_model(stored_model=stored_model, st_container=st_container)
