from __future__ import annotations

import logging
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
from lightgbm import LGBMModel
from matplotlib.figure import Figure
from pyspark.sql import DataFrame
from pyspark.sql.types import NumericType

from faas.base import Passthrough, Pipeline
from faas.encoder import OrdinalEncoder
from faas.scaler import StandardScaler
from faas.utils_dataframe import JoinableByRowID, get_non_numeric_columns

logger = logging.getLogger(__name__)


def is_numeric(df: DataFrame, column: str) -> str:
    return isinstance(df.schema[column].dataType, NumericType)


class E2EPipline:

    def __init__(
        self,
        df: DataFrame,
        target_column: str,
        target_group: Optional[str] = None,  # = 'protocol_type'
        feature_columns: Optional[str] = None
    ):
        self.target_column = target_column
        if feature_columns is None:
            feature_columns = [c for c in df.columns if c != target_column]
            logger.info(
                'Received None as feature_columns, automatically using all non-target columns. '
                f'len(feature_columns): {len(feature_columns)}'
            )
        self.feature_columns = feature_columns

        categorical_features = get_non_numeric_columns(df.select(*feature_columns))
        numeric_features = [c for c in feature_columns if c not in categorical_features]
        logger.info(
            f'num_features: {len(feature_columns)} '
            f'num_categorical_features: {len(categorical_features)} '
            f'num_numeric_features: {len(numeric_features)}'
        )
        self.categorical_features = categorical_features
        self.numeric_features = numeric_features

        xsteps = [Passthrough(columns=self.numeric_features)]
        for c in self.categorical_features:
            xsteps.append(OrdinalEncoder(c))
        self.x_pipeline = Pipeline(steps=xsteps)

        ysteps = []
        if target_group is not None:
            ysteps.append(StandardScaler(column=target_column, group_column=target_group))
        else:
            ysteps.append(Passthrough(columns=[self.target_column]))
        self.y_pipeline = Pipeline(steps=ysteps)

        self.m = LGBMModel(
            objective='regression' if is_numeric(df=df, column=self.target_column) else 'binary',
            deterministic=True,
        )

    def get_x(self, df: DataFrame) -> pd.DataFrame:
        return (
            self.x_pipeline
            .transform(df)
            .select(self.x_pipeline.feature_columns)
            .toPandas()
        )

    def get_y(self, df: DataFrame) -> pd.DataFrame:
        return (
            self.y_pipeline
            .transform(df)
            .select(self.y_pipeline.feature_columns)
            .toPandas()
        )

    def fit(self, df: DataFrame) -> E2EPipline:
        self.x_pipeline.fit(df)
        self.y_pipeline.fit(df)
        X, y = self.get_x(df), self.get_y(df)
        self.m.fit(X=X, y=y)
        return self

    def predict(self, df: DataFrame) -> DataFrame:
        jb = JoinableByRowID(df)
        Xpred = self.get_x(jb.df)
        ypred = self.m.predict(Xpred)
        df_with_y = jb.join_by_row_id(
            ypred,
            column=self.y_pipeline.feature_columns[0]
        )
        df_pred = self.y_pipeline.inverse_transform(df_with_y)
        return df_pred


def plot_feature_importances(m: LGBMModel, top_n: int = 10) -> Figure:
    df = pd.DataFrame({
        'name': m.feature_name_,
        'importance': m.feature_importances_,
    }).sort_values('importance', ascending=False)
    top_df = df.iloc[:top_n]
    with plt.xkcd():
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(x=top_df['name'], height=top_df['importance'])
        ax.tick_params(labelrotation=90)
        ax.set_title('Feature Importances')
    return fig
