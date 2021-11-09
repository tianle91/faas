from __future__ import annotations

import logging

import pyspark.sql.functions as F
from pyspark.sql import DataFrame
from pyspark.sql.types import DoubleType

logger = logging.getLogger(__name__)

ROW_ID_COL = '__ROW_ID__'


class LGBMDataWrapper:
    # def __init__(
    #     self,
    #     target_column: str,
    #     covariate_columns: List[str],
    #     categorical_columns: Optional[List[str]],
    #     target_scaling_group_by_column: Optional[str],
    #     **lgbm_params
    # ) -> None:
    #     self.target_column = target_column
    #     self.covariate_columns = covariate_columns
    #     self.pp = Preprocess(
    #         categorical_columns=categorical_columns,
    #         target_column=target_column,
    #         scale_by_column=target_scaling_group_by_column,
    #     )
    #     self.m = LGBMRegressor(**lgbm_params)

    # def fit(self, df: DataFrame) -> LGBMDataWrapper:
    #     df = df.select(*self.covariate_columns, self.target_column)
    #     # update Preprocess and apply transforms
    #     self.pp.fit(df)
    #     xy = self.pp.transform(df).toPandas()
    #     # train model with X, y
    #     X, y = xy[self.covariate_columns], xy[self.target_column]
    #     self.m.fit(X, y)
    #     return self

    def predict(self, df: DataFrame):
        df = df.select(*self.covariate_columns)
        # insert an index because we need to join predicted results back
        df = df.withColumn(ROW_ID_COL, F.monotonically_increasing_id())
        pdf = self.pp.transform_X(df).toPandas()
        ypreds = self.m.predict(pdf[self.covariate_columns])
        # add target column into df
        ypred_mapping = {i: float(ypred) for i, ypred in zip(pdf[ROW_ID_COL], ypreds)}
        df = df.withColumn(
            self.target_column,
            F.udf(lambda i: ypred_mapping[i], DoubleType())(F.col(ROW_ID_COL))
        )
        # inverse the transformations
        df = self.pp.inverse_transform_X(df)
        df = self.pp.inverse_transform_y(df)
        return df
