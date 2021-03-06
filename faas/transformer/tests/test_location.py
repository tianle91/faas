from pyspark.sql import SparkSession

from faas.transformer.location import OpenRouteServiceFeatures


def test_DayOfWeekFeatures(spark: SparkSession):
    lat, lon = 43.651070, -79.347015
    df = spark.createDataFrame(data=[{'lat': lat, 'lon': lon}])
    res = OpenRouteServiceFeatures('lon', 'lat').fit(df)
    res.transform(df).toPandas()
