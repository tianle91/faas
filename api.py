from typing import List, Optional

import pyspark.sql.functions as F
from fastapi import FastAPI
from pydantic import BaseModel
from pyspark.sql import DataFrame, SparkSession

from faas.lightgbm import LGBMWrapper
from faas.storage import list_models, read_model

app = FastAPI()


@app.get('/')
def read_root():
    return list_models()


@app.get('/models/{model_key}')
def read_item(model_key: str):
    return read_model(model_key)


class PredictionRequest(BaseModel):
    data: List[dict]


class PredictionResponse(BaseModel):
    prediction: Optional[List[dict]] = None
    messages: Optional[List[str]] = None


@app.post('/predict/{model_key}', response_model=PredictionResponse)
async def predict(model_key: str, prediction_request: PredictionRequest) -> PredictionResponse:
    # try to load model
    try:
        m: LGBMWrapper = read_model(model_key)
    except KeyError:
        return PredictionResponse(prediction=[], messages=['Model not found'])

    spark = (
        SparkSession
        .builder
        .appName(f'model_key_{model_key}')
        .getOrCreate()
    )
    df: DataFrame = spark.createDataFrame(data=prediction_request.data)

    # conversion to date
    date_column = m.config.weight.date_column
    if date_column is not None and date_column in df.columns:
        df = df.withColumn(date_column, F.to_date(date_column))

    # check input dataframe
    ok, msgs = m.check_df_prediction(df=df)
    if not ok:
        return PredictionResponse(prediction=[], messages=msgs)
    else:
        df_predict = m.predict(df).toPandas()
        return PredictionResponse(prediction=df_predict.to_dict(orient='records'), messages=msgs)
