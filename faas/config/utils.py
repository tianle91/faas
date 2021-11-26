from typing import List

from pyspark.sql import DataFrame
from pyspark.sql.types import DataType


def get_columns_by_type(df: DataFrame, dtype: DataType) -> List[str]:
    out = []
    for c in df.columns:
        if isinstance(df.schema[c].dataType, dtype):
            out.append(c)
    return out
