from functools import reduce
from typing import List

import pandas as pd
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation
from pyspark.sql import DataFrame
from pyspark.sql.types import NumericType, StringType

from faas.config.utils import get_columns_by_type
from faas.transformer.encoder import OneHotEncoder

__CORRELATION_VECTOR_COL__ = '__CORRELATION_VECTOR_COL__'


def get_arg_max_abs(seq: List[float]) -> float:
    return reduce(lambda a, b: a if abs(a) > abs(b) else b, seq)


def correlation(df: DataFrame, columns: List[str], collapse_categorical=True) -> pd.DataFrame:
    """Return correlation matrix between columns."""

    numeric_columns = get_columns_by_type(df=df, dtype=NumericType)
    categorical_columns = get_columns_by_type(df=df, dtype=StringType)
    for c in columns:
        if c not in numeric_columns and c not in categorical_columns:
            raise TypeError(
                f'Cannot find correlation for column: {c} as it is neither numeric nor string.')

    categorical_column_mapping = {}
    for c in categorical_columns:
        enc = OneHotEncoder(categorical_column=c).fit(df)
        categorical_column_mapping[c] = enc.feature_columns
        df = enc.transform(df).drop(c)

    raw_columns = numeric_columns.copy()
    if len(categorical_column_mapping) > 0:
        raw_columns += reduce(lambda a, b: a + b, categorical_column_mapping.values())

    assembler = VectorAssembler(
        inputCols=raw_columns,
        outputCol=__CORRELATION_VECTOR_COL__
    )
    df = assembler.transform(df.fillna(0.)).select(__CORRELATION_VECTOR_COL__)
    correlation = Correlation.corr(df, __CORRELATION_VECTOR_COL__)

    m = correlation.collect()[0][correlation.columns[0]].toArray()
    corr_pdf = pd.DataFrame(m, index=raw_columns, columns=raw_columns, dtype=float)

    if collapse_categorical:
        for c, v in categorical_column_mapping.items():
            # merge the rows
            corr_pdf.loc[c, :] = pd.concat([
                corr_pdf.loc[[v_c], :] for v_c in v
            ], axis=0).apply(get_arg_max_abs, axis=0)
            corr_pdf = corr_pdf.drop(labels=v)
            # merge the columns
            corr_pdf.loc[:, c] = pd.concat([
                corr_pdf.loc[:, [v_c]] for v_c in v
            ], axis=1).apply(get_arg_max_abs, axis=1)
            corr_pdf = corr_pdf.drop(columns=v)

        # retain order
        corr_pdf = corr_pdf.loc[columns, columns]

    return corr_pdf
