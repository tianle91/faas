from typing import List

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

    def join_by_row_id(self, x: List[float], column: str) -> DataFrame:
        mapping = {i: float(val) for i, val in enumerate(x)}
        udf = F.udf(lambda i: mapping[i], DoubleType())
        return self.df.withColumn(column, udf(F.col(ROW_ID_COL)))
