from __future__ import annotations

import logging
from typing import List, Optional

from pyspark.sql import DataFrame
from pyspark.sql.types import NumericType

from faas.encoder import OrdinalEncoderSingleSpark
from faas.scaler import StandardScalerSpark

logger = logging.getLogger(__name__)


def get_non_numeric_types(df: DataFrame) -> List[str]:
    return [c for c in df.columns if not isinstance(df.schema[c].dataType, NumericType)]


class Preprocess:
    """Preprocess data.

    1. Convert categorical data into ordinal encoding using OrdinalEncoder.
    2. (TBD) Scale target features within each group using StandardScaler.
    3. (TBD) add date-related features.
    4. (TBD) add location-related features.
    """

    def __init__(
        self,
        categorical_columns: Optional[List[str]] = None,
        target_scaling_group_by_column: Optional[str] = None,
        target_column: Optional[str] = None
    ):
        self.encoder = {c: OrdinalEncoderSingleSpark(c) for c in categorical_columns}
        if target_column is None and target_scaling_group_by_column is not None:
            raise ValueError(
                f'Cannot specify '
                f'target_scaling_group_by_column: {target_scaling_group_by_column} '
                f'without specifying target_column.'
            )
        self.scaler = None
        if target_column is not None:
            self.scaler = StandardScalerSpark(
                column=target_column, group_column=target_scaling_group_by_column)

    def fit(self, df: DataFrame) -> Preprocess:
        if self.scaler is not None:
            self.scaler.fit(df)
        for enc in self.encoder.values():
            enc.fit(df)
        return self

    def transform(self, df: DataFrame) -> DataFrame:
        if self.scaler is not None:
            self.scaler.transform(df)
        for enc in self.encoder.values():
            df = enc.transform(df)
        return df

    def inverse_transform(self, df: DataFrame) -> DataFrame:
        for enc in self.encoder.values():
            df = enc.inverse_transform(df)
        if self.scaler is not None:
            self.scaler.inverse_transform(df)
        return df
