from typing import List

from pyspark.sql import DataFrame

from faas.base import BaseTransformer


class SeasonalFeatures(BaseTransformer):

    def __init__(
        self,
        date_column: str,
        day_of_week: bool = True,
        day_of_month: bool = True,
        day_of_year: bool = True,
        week_of_year: bool = True,
    ) -> None:
        self.date_column = date_column
        self.day_of_week = day_of_week
        self.day_of_month = day_of_month
        self.day_of_year = day_of_year
        self.week_of_year = week_of_year

    @property
    def feature_columns(self) -> List[str]:
        return super().feature_columns

    def transform(self, df: DataFrame) -> DataFrame:
        pass
