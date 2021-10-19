import logging

from pyspark.sql import DataFrame

from faas.encoder import OrdinalEncoderSingleSpark

logger = logging.getLogger(__name__)


class Preprocess:
    """Preprocess data.

    1. Convert categorical data into ordinal encoding using OrdinalEncoder.
    2. (TBD) Scale target features within each group using StandardScaler.
    3. (TBD) add date-related features.
    4. (TBD) add location-related features.
    """

    def __init__(self, categorical_columns):
        self.encoder = {c: OrdinalEncoderSingleSpark(c) for c in categorical_columns}

    def fit(self, df: DataFrame):
        for enc in self.encoder.values():
            enc.fit(df)

    def transform(self, df: DataFrame) -> DataFrame:
        for enc in self.encoder.values():
            df = enc.transform(df)
        return df
