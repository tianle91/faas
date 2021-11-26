import json
from typing import List, Optional

import pyspark.sql.functions as F
from fastapi import FastAPI
from pydantic import BaseModel
from pyspark.sql import DataFrame, SparkSession

from faas.storage import list_models, read_model, set_num_calls_remaining

app = FastAPI()

spark = (
    SparkSession
    .builder
    .appName('api_predict')
    .getOrCreate()
)


@app.get('/')
def root():
    return json.dumps([str(k) for k in list_models()])


class PredictionRequest(BaseModel):
    model_key: str
    data: List[dict]


class PredictionResponse(BaseModel):
    prediction: Optional[List[dict]] = None
    messages: Optional[List[str]] = None
    num_calls_remaining: Optional[int] = None


@app.post('/predict', response_model=PredictionResponse)
async def predict(prediction_request: PredictionRequest) -> PredictionResponse:
    model_key = prediction_request.model_key
    try:
        stored_model = read_model(model_key)
    except KeyError:
        return PredictionResponse(messages=['Model not found'])

    if stored_model.num_calls_remaining <= 0:
        return PredictionResponse(messages=[
            f'Insufficient num_calls_remaining: {stored_model.num_calls_remaining}'
        ])

    # load the requested dataframe
    df: DataFrame = spark.createDataFrame(data=prediction_request.data)
    conf = stored_model.config

    if conf.date_column is not None:
        df = df.withColumn(
            conf.date_column,
            F.to_date(conf.date_column, conf.date_column_format)
        )

    m = stored_model.m
    ok, msgs = m.check_df_prediction(df=df)

    if not ok:
        return PredictionResponse(messages=msgs)

    df_predict = m.predict(df).toPandas()
    new_num_calls_remaining = stored_model.num_calls_remaining - 1
    set_num_calls_remaining(key=model_key, n=new_num_calls_remaining)
    return PredictionResponse(
        prediction=df_predict.to_dict(orient='records'),
        messages=msgs,
        num_calls_remaining=new_num_calls_remaining,
    )
