from typing import List

import pandas as pd
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation
from pyspark.sql import DataFrame
from pyspark.sql.types import NumericType

__CORRELATION_VECTOR_COL__ = "__CORRELATION_VECTOR_COL__"


def correlation(
    df: DataFrame,
    feature_columns: List[str],
    target_column: str
) -> pd.DataFrame:
    columns = [
        c for c in feature_columns + [target_column]
        if isinstance(df.schema[c].dataType, NumericType)
    ]
    assembler = VectorAssembler(
        inputCols=columns,
        outputCol=__CORRELATION_VECTOR_COL__
    )
    df = assembler.transform(df).select(__CORRELATION_VECTOR_COL__)
    correlation = Correlation.corr(df, __CORRELATION_VECTOR_COL__)
    m = correlation.collect()[0][correlation.columns[0]].toArray()
    corr_df = pd.DataFrame(m, index=columns, columns=columns, dtype=float)
    return corr_df
