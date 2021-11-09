from __future__ import annotations

from typing import List

from pyspark.sql import DataFrame


class BaseTransformer:
    """A BaseTransformer adds a single column, i.e. feature_column"""

    @property
    def feature_columns(self) -> List[str]:
        raise NotImplementedError

    def fit(self, df: DataFrame) -> BaseTransformer:
        return self

    def transform(self, df: DataFrame) -> DataFrame:
        return df


class InvertibleTransformer(BaseTransformer):

    def inverse_transform(self, df: DataFrame) -> DataFrame:
        return df


class Passthrough(InvertibleTransformer):

    def __init__(self, columns: str) -> None:
        self.columns = columns

    @property
    def feature_columns(self) -> List[str]:
        return self.columns


class Pipeline(BaseTransformer):
    def __init__(self, steps: List[BaseTransformer]):
        self.steps = steps

    @property
    def feature_columns(self) -> List[str]:
        res = []
        for transformer in self.steps:
            res += transformer.feature_columns
        return res

    def fit(self, df: DataFrame) -> Pipeline:
        for transformer in self.steps:
            transformer.fit(df)
        return self

    def transform(self, df: DataFrame) -> DataFrame:
        for transformer in self.steps:
            df = transformer.transform(df)
        return df

    def inverse_transform(self, df: DataFrame) -> DataFrame:
        for transformer in self.steps[::-1]:
            if not isinstance(transformer, InvertibleTransformer):
                raise TypeError(
                    f'Transformer {transformer} should be InvertibleTransformer')
            transformer: InvertibleTransformer = transformer
            df = transformer.inverse_transform(df)
        return df
