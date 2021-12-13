from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import pyspark.sql.functions as F
from lightgbm import LGBMClassifier, LGBMRegressor
from pyspark.sql import DataFrame

from faas.transformer.etl import (ETLConfig, WTransformer, XTransformer,
                                  YCategoricalTransformer, YNumericTransformer,
                                  merge_validations)
from faas.utils.dataframe import JoinableByRowID

logger = logging.getLogger(__name__)

TEMP_Y_COLUMN = '__TEMP_Y_COLUMN__'


class ETLWrapperForLGBM:

    def __init__(self, config: ETLConfig):
        self.config = config
        self.xtransformer = XTransformer(config.feature)
        self.wtransformer = WTransformer(config.weight)
        if config.target.is_categorical:
            self.ytransformer = YCategoricalTransformer(config.target)
        else:
            self.ytransformer = YNumericTransformer(config.target)

    def check_df_prediction(self, df: DataFrame) -> Tuple[bool, List[str]]:
        # TODO: check that all columns exist prior to validating
        return merge_validations([
            self.xtransformer.validate_input(df=df),
            self.ytransformer.validate_input(df=df, prediction=True),
        ])

    def check_df_train(self, df: DataFrame) -> Tuple[bool, List[str]]:
        validations = [
            self.xtransformer.validate_input(df=df),
            self.ytransformer.validate_input(df=df, prediction=False),
        ]
        if self.wtransformer is not None:
            validations.append(self.wtransformer.validate_input(df=df))
        return merge_validations(validations)

    def fit(self, df: DataFrame) -> ETLWrapperForLGBM:
        ok, msgs = self.check_df_train(df)
        if not ok:
            raise ValueError(msgs)
        # get the matrices
        X = self.xtransformer.fit(df).get_transformed_as_pdf(df)
        logger.info(f'X.shape: {X.shape}')
        y = self.ytransformer.fit(df).get_transformed_as_pdf(df)
        logger.info(f'y.shape: {y.shape}')
        p = {}
        if self.wtransformer is not None:
            w = self.wtransformer.fit(df).get_transformed_as_pdf(df)
            logger.info(f'w.shape: {w.shape}')
            if w.shape[1] != 1:
                raise ValueError('There should be only a single column in w')
            p['sample_weight'] = w.iloc[:, 0]
        # model params
        lgbm_params = {'deterministic': True}
        if self.config.target.is_categorical:
            num_class = self.ytransformer.num_classes
            # lightgbm.basic.LightGBMError: Number of classes must be 1 for non-multiclass training
            lgbm_params['num_class'] = num_class if num_class > 2 else 1
            # LGBMModel.predict returns probabilities instead of label
            self.m = LGBMClassifier(**lgbm_params)
        else:
            self.m = LGBMRegressor(**lgbm_params)
        # fit
        feature_name = self.xtransformer.feature_columns
        categorical_feature = self.xtransformer.encoded_categorical_feature_columns
        self.m.fit(X=X, y=y, feature_name=feature_name,
                   categorical_feature=categorical_feature, **p)
        return self

    def predict(self, df: DataFrame, output_column: Optional[str] = None) -> DataFrame:
        ok, msgs = self.check_df_prediction(df)
        if not ok:
            raise ValueError(msgs)

        target_col = self.ytransformer.feature_columns[0]

        # temporarily store the actual target in some other column
        store_target_temporarily = output_column is not None and target_col in df.columns
        if store_target_temporarily:
            df = df.withColumn(TEMP_Y_COLUMN, F.col(target_col))

        # ensure rows are identifiable
        jb = JoinableByRowID(df)
        # get the matrices and predict
        Xpred = self.xtransformer.get_transformed_as_pdf(jb.df)
        ypred = self.m.predict(Xpred)

        # join them back to df
        df_with_y = jb.join_by_row_id(ypred, column=target_col)
        df_pred = self.ytransformer.inverse_transform(df_with_y)

        if output_column is not None:
            df_pred = df_pred.withColumn(output_column, F.col(target_col))
            if output_column != target_col and store_target_temporarily:
                # then we can reinstate the original column
                df_pred = df_pred.withColumn(target_col, F.col(TEMP_Y_COLUMN)).drop(TEMP_Y_COLUMN)

        return df_pred
