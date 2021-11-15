from dataclasses import dataclass
from typing import List, Optional

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import (DateType, DoubleType, StringType, StructField,
                               StructType)

from faas.utils.dataframe import (get_date_columns, get_non_numeric_columns,
                                  get_numeric_columns)


@dataclass
class CSVTypes:
    numeric_columns: Optional[List[str]] = None,
    categorical_columns: Optional[List[str]] = None,
    date_columns: Optional[List[str]] = None,
    date_format: str = 'yyyy-MM-dd'

    @property
    def validate(self) -> bool:
        if self.numeric_columns is None and self.categorical_columns is None:
            raise ValueError('One of numeric or categorical must be set.')


def load_csv_with_types(spark: SparkSession, p: str, csv_type: CSVTypes) -> DataFrame:
    fields = []
    csv_type.validate()
    if csv_type.numeric_columns is not None:
        fields += [
            fields.append(StructField(c, DoubleType(), nullable=True))
            for c in csv_type.numeric_columns
        ]
    if csv_type.categorical_columns is not None:
        fields += [
            fields.append(StructField(c, StringType(), nullable=True))
            for c in csv_type.categorical_columns
        ]
    if csv_type.date_columns is not None:
        fields += [
            fields.append(StructField(c, DateType(), nullable=True))
            for c in csv_type.numeric_columns
        ]
    df = (
        spark
        .read
        .format('csv')
        .options(header=True, inferSchema=False, dateFormat=csv_type.date_format)
        .schema(StructType(fields=fields))
        .load(p)
    )
    return df


def inferred_types(df: DataFrame) -> CSVTypes:
    return CSVTypes(
        numeric_columns=get_numeric_columns(df),
        categorical_columns=get_non_numeric_columns(df),
        date_columns=get_date_columns(df),
    )
