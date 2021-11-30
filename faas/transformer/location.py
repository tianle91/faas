import os
from typing import List

import openrouteservice
import pyspark.sql.functions as F
from geopy import Point
from pyspark.sql import DataFrame
from pyspark.sql.types import DoubleType

from faas.transformer.base import BaseTransformer

CLIENT = openrouteservice.Client(key=os.getenv('ORS_SECRET'))
ISOCHRONE_SECONDS: List[float] = [60, 600, 1800, 3600]
ISOCHRONE_ATTRIBUTES: List[str] = ['area', 'reachfactor', 'total_pop']


def get_value_from_isochrone_res(res: dict, i: int, isochrone_sec: float, isochrone_attr: str):
    properties_dict = res['features'][i]['properties']
    actual_seconds = properties_dict['value']
    if actual_seconds != isochrone_sec:
        raise ValueError(
            f'res.features[{i}].properties.value: {actual_seconds} '
            f'should be {isochrone_sec}!'
        )
    return properties_dict[isochrone_attr]


GET_VALUE_FROM_ISOCHRONE_RES_PARAMS = {
    f'OpenRouteServiceFeatures_{isochrone_attr}_{isochrone_sec}': {
        'i': i,
        'isochrone_sec': isochrone_sec,
        'isochrone_attr': isochrone_attr
    }
    for isochrone_attr in ISOCHRONE_ATTRIBUTES
    for i, isochrone_sec in enumerate(ISOCHRONE_SECONDS)
}


class OpenRouteServiceFeatures(BaseTransformer):
    def __init__(self, longitude_column: str, latitude_column: str):
        self.longitude_column = longitude_column
        self.latitude_column = latitude_column
        self.mapping = {}

    @property
    def input_columns(self) -> List[str]:
        raise [self.longitude_column, self.latitude_column]

    @property
    def feature_columns(self) -> List[str]:
        return list(GET_VALUE_FROM_ISOCHRONE_RES_PARAMS.keys())

    def fit(self, df: DataFrame) -> BaseTransformer:
        for row in df.select(self.latitude_column, self.longitude_column).distinct().collect():
            lat, lon = row[self.latitude_column], row[self.longitude_column]
            Point(latitude=lat, longitude=lon)  # validate lat lon
            latlon = (lat, lon)
            if latlon not in self.mapping:
                res = CLIENT.isochrones(
                    # OpenRouteService takes longitude, latitude
                    locations=[(lon, lat), ],
                    attributes=ISOCHRONE_ATTRIBUTES,
                    range=ISOCHRONE_SECONDS,
                )
                self.mapping[latlon] = {
                    k: get_value_from_isochrone_res(
                        res=res, **GET_VALUE_FROM_ISOCHRONE_RES_PARAMS[k])
                    for k in GET_VALUE_FROM_ISOCHRONE_RES_PARAMS
                }
        return self

    def transform(self, df: DataFrame) -> DataFrame:
        for c in self.feature_columns:
            udf = F.udf(lambda lat, lon: self.mapping[(lat, lon)][c], DoubleType())
            df = df.withColumn(c, udf(self.latitude_column, self.longitude_column))
        return df
