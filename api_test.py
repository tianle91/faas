import json
import os
from datetime import datetime

import pandas as pd
import pyspark.sql.functions as F
import requests
from pyspark.sql import SparkSession

from faas.config import Config
from faas.config.config import create_etl_config
from faas.lightgbm import ETLWrapperForLGBM
from faas.storage import StoredModel, write_model

APIURL = os.getenv('APIURL', default='http://localhost:8000')
spark = SparkSession.builder.appName('api_test.py').getOrCreate()
p = 'data/sample_multi_ts.csv'

# load training data
df = spark.read.options(header=True, inferSchema=True).csv(p)
df = df.withColumn('date', F.to_date('date'))

# do some training
conf = Config(
    target='numeric_1',
    date_column='date',
    group_columns=['ts_type'],
    feature_columns=['categorical_0', 'numeric_0'],
)
m = ETLWrapperForLGBM(config=create_etl_config(conf=conf, df=df)).fit(df)

# write to storage
stored_model = StoredModel(
    dt=datetime.now(),
    m=m,
    config=conf
)
model_key = write_model(stored_model)

# get prediction
pred_pdf = pd.read_csv(p).head().drop(columns=['numeric_1'])
r = requests.post(
    url=f'{APIURL}/predict',
    data=json.dumps({
        'model_key': model_key,
        'data': pred_pdf.to_dict(orient='records'),
    })
)
response_json = r.json()
pred_pdf_received = pd.DataFrame(response_json['prediction'])

# tests
if len(pred_pdf) != len(pred_pdf_received):
    raise ValueError(f'{len(pred_pdf):} != {len(pred_pdf_received):}')
if 'numeric_1' not in pred_pdf_received.columns:
    raise KeyError(f'numeric_1 not in {pred_pdf_received.columns:}')
