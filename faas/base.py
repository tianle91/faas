from __future__ import annotations

from pyspark.sql import DataFrame


class BaseTransformer:
    """A BaseTransformer adds a single column, i.e. feature_column"""

    @property
    def feature_column(self) -> str:
        raise NotImplementedError

    def fit(self) -> BaseTransformer:
        return self

    def transform(self, df: DataFrame) -> DataFrame:
        raise NotImplementedError


class XTransformer(BaseTransformer):
    pass


class WTransformer(BaseTransformer):
    pass


class YTransformer(BaseTransformer):
    """YTransformers need to implement inverse"""

    def inverse_transform(self, df: DataFrame) -> DataFrame:
        raise NotImplementedError
