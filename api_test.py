import json

import pandas as pd
import pyspark.sql.functions as F
import requests
from pyspark.sql import SparkSession

from faas.config import Config, FeatureConfig, TargetConfig, WeightConfig
from faas.lightgbm import LGBMWrapper
from faas.storage import write_model

API_URL = 'http://localhost:8000'
p = 'data/sample_multi_ts.csv'

# load training data
spark = SparkSession.builder.getOrCreate()
df = spark.read.options(header=True, inferSchema=True).csv(p)
df = df.withColumn('date', F.to_date('date'))

# do some training
config = Config(
    feature=FeatureConfig(
        categorical_columns=['categorical_0'],
        numeric_columns=['numeric_0'],
        date_column='date'
    ),
    target=TargetConfig(
        column='numeric_1',
    ),
    weight=WeightConfig(
        date_column='date',
        group_columns=['ts_type'],
    )
)
model_key = write_model(LGBMWrapper(config=config).fit(df))


# get prediction
pred_pdf = pd.read_csv(p).head().drop(columns=['numeric_1'])
r = requests.post(
    url=f'{API_URL}/predict/{model_key}',
    data=json.dumps({'data': pred_pdf.to_dict(orient='records')})
)
response_json = r.json()
pred_pdf_received = pd.DataFrame(response_json['prediction'])


# test prediction df
assert len(pred_pdf) == len(pred_pdf_received)
assert 'numeric_1' in pred_pdf_received.columns
