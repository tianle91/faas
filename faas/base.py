from __future__ import annotations

from typing import List, Tuple

from pyspark.sql import DataFrame


class BaseTransformer:
    """A BaseTransformer adds a single column, i.e. feature_column"""

    @property
    def feature_column(self) -> str:
        raise NotImplementedError

    def fit(self, df: DataFrame) -> BaseTransformer:
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


class Pipeline(BaseTransformer):
    def __init__(self, steps: List[Tuple[str, BaseTransformer]]):
        self.steps = steps

    @property
    def feature_columns(self) -> List[str]:
        return [transformer.feature_column for _, transformer in self.steps]

    def fit(self, df: DataFrame) -> Pipeline:
        for _, transformer in self.steps:
            transformer.fit(df)
        return self

    def transform(self, df: DataFrame) -> DataFrame:
        for _, transformer in self.steps:
            df = transformer.transform(df)
        return df

    def inverse_transform(self, df: DataFrame) -> DataFrame:
        for name, transformer in self.steps[::-1]:
            if not isinstance(transformer, YTransformer):
                raise TypeError(
                    f'Transformer name: {name}, {transformer} should be a YTransformer '
                    'in order to be invertible.'
                )
            transformer: YTransformer = transformer
            df = transformer.inverse_transform(df)
        return df
