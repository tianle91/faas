from __future__ import annotations

from typing import List

import pandas as pd
from pyspark.sql import DataFrame


class BaseTransformer:
    """A BaseTransformer expect input_columns and adds feature_columns."""

    @property
    def input_columns(self) -> List[str]:
        raise NotImplementedError

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

    def __init__(self, columns: List[str]) -> None:
        self.columns = columns

    @property
    def input_columns(self) -> List[str]:
        return self.columns

    @property
    def feature_columns(self) -> List[str]:
        return self.columns


class Pipeline(BaseTransformer):
    def __init__(self, steps: List[BaseTransformer]):
        self.steps = steps

    @property
    def input_columns(self) -> List[str]:
        in_cols = []
        out_cols = []
        for transformer in self.steps:
            in_cols += [c for c in transformer.input_columns if c in out_cols]
            out_cols += transformer.feature_columns
        return in_cols

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

    def get_transformed_as_pdf(self, df: DataFrame) -> pd.DataFrame:
        return self.transform(df).select(*self.feature_columns).toPandas()

    def inverse_transform(self, df: DataFrame) -> DataFrame:
        for transformer in self.steps[::-1]:
            if not isinstance(transformer, InvertibleTransformer):
                raise TypeError(
                    f'Transformer {transformer} should be InvertibleTransformer')
            transformer: InvertibleTransformer = transformer
            df = transformer.inverse_transform(df)
        return df
