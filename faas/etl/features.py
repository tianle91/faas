import os
from typing import List, Optional

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from tktransformers import PassThrough
from tktransformers.datetime.seasonality import (SEASONALITY_FEATURE_MAPPING,
                                                 SeasonalityFeature)
from tktransformers.location.openrouteservice import OpenRouteServiceFeature


def get_features_transformer(
    categorical_columns: List[str],
    numeric_columns: List[str],
    date_column: Optional[str] = None,
    latitude_column: Optional[str] = None,
    longitude_column: Optional[str] = None,
) -> ColumnTransformer:
    transformers = []
    if len(numeric_columns) > 0:
        transformers.append(('PassThrough', PassThrough(), numeric_columns))
    for c in categorical_columns:
        transformers.append((f'OrdinalEncoder_{c}', OrdinalEncoder(), c))
    if date_column is not None:
        for seasonality_name in SEASONALITY_FEATURE_MAPPING:
            transformers.append((
                f'SeasonalityFeature_{seasonality_name}',
                SeasonalityFeature(seasonality_name=seasonality_name),
                date_column
            ))
    if latitude_column is not None and longitude_column is not None:
        transformers.append((
            'OpenRouteServiceFeature',
            OpenRouteServiceFeature(key=os.getenv('ORS_SECRET')),
            [latitude_column, longitude_column]
        ))
    return ColumnTransformer(transformers=transformers)
