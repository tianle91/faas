import streamlit as st

from ui import pages

st.sidebar.title('FaaS 🌲')
st.sidebar.markdown('To start, head to Training.')

page_selector = st.sidebar.radio(label='page', options=pages.keys())
if page_selector is not None:
    pages[page_selector]()
