import streamlit as st
from pyspark.sql import SparkSession

from faas.storage import read_model
from faas.utils.io import load_cached_df_from_st_uploaded
from ui import pages


def get_spark():
    spark = (
        SparkSession
        .builder
        .appName('ui')
        .config('spark.driver.maxResultsSize', '16g')
        .config('spark.driver.memory', '16g')
        .getOrCreate()
    )
    return spark


@st.cache(max_entries=1)
def update_state_with_new_upload(f):
    spark = get_spark()
    df = load_cached_df_from_st_uploaded(f=f, spark=spark)
    st.session_state['df'] = df
    with st.spinner('Profiling dataframe...'):
        summary_pdf = df.describe().toPandas().set_index(keys='summary')
    st.session_state['summary_pdf'] = summary_pdf
    st.success(f'Ingested {df.count()} rows')


st.sidebar.title('FaaS 🌲')

# populate states: df, summary_df
f = st.sidebar.file_uploader('Upload dataframe', type=['csv', 'parquet'])
if f is not None:
    update_state_with_new_upload(f)

# populate states: model_key, stored_model
model_key = st.session_state.get('model_key', '')
model_key_input = st.sidebar.text_input('Model key (obtain this from training)', value=model_key)
if model_key_input != '':
    try:
        st.session_state['model_key'] = model_key_input
        st.session_state['stored_model'] = read_model(key=st.session_state['model_key'])
        st.sidebar.success('Model loaded')
    except Exception as e:
        st.sidebar.error(e)

# set up the pages
page_selector = st.sidebar.radio(label='Pages', options=pages.keys())
if page_selector is not None:
    pages[page_selector]()
