from datetime import date, timedelta
from typing import Dict, List, Tuple

from numpy.random import default_rng

ALPHABET = [c for c in 'ABCDEFGHIJLKMNOPQRSTUVWXYZ']


class GenerateSynthetic:
    """Generate synthetic data.
    """

    def __init__(self, num_categorical: int = 1, num_numeric: int = 1):
        self.rng = default_rng()
        self.num_categorical = num_categorical
        self.num_numeric = num_numeric

    @property
    def categorical_names(self) -> List[str]:
        return [f'categorical_{i}' for i in range(self.num_categorical)]

    @property
    def numeric_names(self) -> List[str]:
        return [f'numeric_{i}' for i in range(self.num_numeric)]

    def generate_iid(self, n: int = 1000) -> Dict[str, list]:
        res = {}
        for c in self.categorical_names:
            res[c] = self.rng.choice(ALPHABET, size=n)
        for c in self.numeric_names:
            res[c] = self.rng.standard_normal(size=n)
        return res

    def generate_ts(
        self,
        num_days: int = 750,
        date_column: str = 'date'
    ) -> Dict[str, list]:
        res = self.generate_iid(n=num_days)
        res[date_column] = [date(2000, 1, 1) + timedelta(days=i) for i in range(num_days)]
        return res

    def generate_multi_ts(
        self,
        ts_types: List[str],
        ts_type_column: str = 'ts_type',
        num_days: int = 750,
        date_column: str = 'date'
    ) -> Dict[str, list]:
        num_ts = len(ts_types)
        res = self.generate_iid(n=num_ts * num_days)
        res[date_column] = [date(2000, 1, 1) + timedelta(days=i) for i in range(num_days)] * num_ts
        res[ts_type_column] = []
        for ts_type in ts_types:
            res[ts_type_column] += [ts_type, ] * num_days
        return res
