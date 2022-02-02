from datetime import datetime, timedelta
from typing import List, Optional

from sklearn.compose import ColumnTransformer
from tktransformers.weight.decay import WeightDecay
from tktransformers.weight.normalize import WeightNormalize


def get_days_since_epoch(dt: datetime) -> float:
    seconds = dt.strftime('%s')  # seconds since epoch
    return int(seconds) / 86400  # days since epoch


def get_weights_transformer(
    date_column: Optional[str] = None,
    group_columns: Optional[List[str]] = None,
) -> ColumnTransformer:
    transformers = []
    if date_column is not None:
        transformers += [
            (f'WeightNormalize_date_{date_column}', WeightNormalize(), date_column),
            (f'WeightDecay_{date_column}', WeightDecay(half_life=timedelta(days=180)), date_column),
        ]
    if group_columns is not None:
        for c in group_columns:
            transformers.append((f'WeightNormalize_date_{c}', WeightNormalize(), c))
    return ColumnTransformer(transformers=transformers)
