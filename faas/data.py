from typing import List, Tuple

import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

# TODO: convert to spark


def validate_df(
    df: pd.DataFrame,
    covariate_columns: List[str],
    categorical_columns: List[str],
    target_column: str
):
    err_msgs = []
    # categorical columns must not be numeric
    for c in categorical_columns:
        if is_numeric_dtype(df.dtypes[c]):
            err_msgs.append(f'Categorical column {c} is numeric type: {df.dtypes[c]}')
    # covariate columns and target column must be in dataframe
    missing_covariate_columns = set(covariate_columns) - set(df.columns)
    if not len(missing_covariate_columns) == 0:
        err_msgs.append(f'Missing covariate columns: {missing_covariate_columns}')
    if target_column not in df.columns:
        err_msgs.append(f'Missing target column: {target_column}')
    # raise message with all failures
    if len(err_msgs) > 0:
        raise ValueError('\n'.join(err_msgs))


class Preprocess:
    """Preprocess data for lightgbm.
    1. Convert categorical data into ordinal encoding using OrdinalEncoder.
    2. Scale target features within each group using StandardScaler.
    3. (TBD) add date-related features.
    4. (TBD) add location-related features.
    """

    def __init__(self, categorical_columns, target_column) -> None:
        self.categorical_columns = categorical_columns
        self.target_column = target_column
        self._covariate_columns = None

    def validate_df(self, df: pd.DataFrame):
        validate_df(
            df=df,
            covariate_columns=self._covariate_columns,
            categorical_columns=self.categorical_columns,
            target_column=self.target_column
        )

    def transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        self.validate_df(df)
        pass
