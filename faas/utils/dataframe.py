from typing import List, Union

import pandas as pd
import pyspark.sql.functions as F
from pyspark.sql import DataFrame
from pyspark.sql.types import DoubleType

ROW_ID_COL = '__ROW_ID__'


class JoinableByRowID:
    """Adds a row id column and makes a dataframe joinable by row id."""

    def __init__(self, df: DataFrame):
        self.df = (
            df
            .withColumn(ROW_ID_COL, F.monotonically_increasing_id())
            .orderBy(ROW_ID_COL)
        )
        self.ids = [row[ROW_ID_COL] for row in self.df.select(ROW_ID_COL).collect()]

    def join_by_row_id(self, x: List[float], column: str) -> DataFrame:
        if len(x) != len(self.ids):
            raise ValueError(f'len(x) should be {len(self.ids)} but received {len(x)} instead.')
        mapping = {self.ids[i]: float(val) for i, val in enumerate(x)}
        udf = F.udf(lambda i: mapping[i], DoubleType())
        return self.df.withColumn(column, udf(F.col(ROW_ID_COL))).drop(ROW_ID_COL)


def has_duplicates(df: Union[DataFrame, pd.DataFrame]) -> bool:
    if isinstance(df, DataFrame):
        return df.distinct().count() > df.count()
    elif isinstance(df, pd.DataFrame):
        return len(df.drop_duplicates()) < len(df)
