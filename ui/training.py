import pprint as pp
from datetime import datetime

import streamlit as st

from faas.helper import get_trained
from faas.storage import StoredModel, write_model
from ui.config import get_config
from ui.visualization.eda import run_eda
from ui.visualization.vis_df import preview_df
from ui.visualization.vis_model import vis_stored_model


def run_training():

    st.title('Training')
    st.markdown(
        'Upload a dataset with target and feature columns in order to train a model.')

    df = st.session_state.get('df', None)

    if df is not None:
        st.header('Uploaded dataset')
        preview_df(df=df)
        conf = get_config(df=df)

        st.header('Current configuration')
        st.code(pp.pformat(conf.__dict__, compact=True))
        with st.expander('Exploratory Data Analysis'):
            run_eda(conf=conf, df=df)

        if st.button('Train'):

            with st.spinner('Running training...'):
                m = get_trained(conf=conf, df=df)
                stored_model = StoredModel(dt=datetime.now(), m=m, config=conf)

            # write fitted model and store model key in user session
            model_key = write_model(stored_model)
            st.session_state['model_key'] = model_key
            st.session_state['stored_model'] = stored_model

            st.success(
                'Model trained! Model key can be now be used for Prediction.')
            st.markdown(f'Model key: `{model_key}`')

            st.header('Trained model')
            vis_stored_model(stored_model=stored_model)
