from typing import List, Optional

import pyspark.sql.functions as F
from fastapi import FastAPI
from pydantic import BaseModel
from pyspark.sql import DataFrame, SparkSession

from faas.storage import read_model

app = FastAPI()

spark = (
    SparkSession
    .builder
    .appName('api_predict')
    .getOrCreate()
)


class PredictionRequest(BaseModel):
    model_key: str
    data: List[dict]


class PredictionResponse(BaseModel):
    prediction: Optional[List[dict]] = None
    messages: Optional[List[str]] = None


@app.post('/predict', response_model=PredictionResponse)
async def predict(prediction_request: PredictionRequest) -> PredictionResponse:
    # try to load model
    try:
        m, conf = read_model(prediction_request.model_key)
    except KeyError:
        return PredictionResponse(prediction=None, messages=['Model not found'])

    df: DataFrame = spark.createDataFrame(data=prediction_request.data)
    if conf.date_column is not None:
        df = df.withColumn(
            conf.date_column,
            F.to_date(conf.date_column, conf.date_column_format)
        )

    # check input dataframe
    ok, msgs = m.check_df_prediction(df=df)
    if not ok:
        return PredictionResponse(prediction=None, messages=msgs)
    else:
        df_predict = m.predict(df).toPandas()
        return PredictionResponse(prediction=df_predict.to_dict(orient='records'), messages=msgs)
