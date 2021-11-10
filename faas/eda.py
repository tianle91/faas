from typing import List

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure
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
    if target_column not in columns:
        raise TypeError(f'Target: {columns} should be numeric.')
    assembler = VectorAssembler(
        inputCols=columns,
        outputCol=__CORRELATION_VECTOR_COL__
    )
    df = assembler.transform(df).select(__CORRELATION_VECTOR_COL__)
    correlation = Correlation.corr(df, __CORRELATION_VECTOR_COL__)
    m = correlation.collect()[0][correlation.columns[0]].toArray()
    return pd.DataFrame(m, index=columns, columns=columns, dtype=float)


def plot_target_correlation(corr_df: pd.DataFrame, target_column: str, top_n: int = 10) -> Figure:

    corr_df = corr_df.copy()
    corr_df = corr_df[[target_column]].drop(labels=[target_column])
    corr_df['abs_corr'] = corr_df[target_column].apply(lambda v: v if v > 0 else -v)
    corr_df = corr_df.sort_values(by='abs_corr', ascending=False).reset_index()

    with plt.xkcd():
        fig, ax = plt.subplots(figsize=(8, 4))
        top_corr = corr_df.iloc[:top_n]
        ax.bar(x=top_corr['index'], height=top_corr[target_column])
        ax.tick_params(labelrotation=90)
        ax.set_title(f'Top {top_n}')
    return fig
