import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder
from tktransformers import PassThrough


def get_numeric_target_transformer(column: str, log_transform: bool = False) -> ColumnTransformer:
    transformers = []
    if log_transform:
        transformers.append((
            'LogTransform',
            FunctionTransformer(
                lambda x: np.log(1. + x),
                inverse_func=lambda x: np.exp(x) - 1.,
            ),
            column
        ))
    else:
        transformers.append(('PassThrough', PassThrough, column))
    return ColumnTransformer(transformers=transformers)


def get_categorical_target_transformer(column: str) -> ColumnTransformer:
    transformers = [('OrdinalEncoder', OrdinalEncoder(), column)]
    return ColumnTransformer(transformers=transformers)
