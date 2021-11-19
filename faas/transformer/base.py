from __future__ import annotations

from typing import List

import pyspark.sql.functions as F
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
        raise NotImplementedError

    def inverse_transform(self, df: DataFrame) -> DataFrame:
        raise NotImplementedError


class Passthrough(BaseTransformer):

    def __init__(self, columns: List[str]) -> None:
        self.columns = columns

    @property
    def input_columns(self) -> List[str]:
        return self.columns

    @property
    def feature_columns(self) -> List[str]:
        return self.columns

    def transform(self, df: DataFrame) -> DataFrame:
        return df

    def inverse_transform(self, df: DataFrame) -> DataFrame:
        return df


class AddTransformer(BaseTransformer):
    def __init__(self, columns: List[str]) -> None:
        self.columns = columns

    @property
    def input_columns(self) -> List[str]:
        return self.columns

    @property
    def feature_column(self) -> List[str]:
        return f'AddTransformer_{"+".join(self.columns)}'

    @property
    def feature_columns(self) -> List[str]:
        return [self.feature_column]

    def transform(self, df: DataFrame) -> DataFrame:
        res_col = None
        for c in self.columns:
            if res_col is None:
                res_col = F.col(c)
            else:
                res_col += F.col(c)
        return df.withColumn(self.feature_column, res_col)


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

    def inverse_transform(self, df: DataFrame) -> DataFrame:
        for transformer in self.steps[::-1]:
            df = transformer.inverse_transform(df)
        return df
