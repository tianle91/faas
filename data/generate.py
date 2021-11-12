from datetime import date, timedelta
from typing import Dict, List

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
        return [f'categorical_{i}' for i in self.num_categorical]

    @property
    def numeric_names(self) -> List[str]:
        return [f'numeric_{i}' for i in self.num_numeric]

    def generate_iid(self, n: int = 1000) -> Dict[str, list]:
        res = {}
        for c in self.categorical_names:
            res[c] = self.rng.choice(ALPHABET, size=n)
        for c in self.num_numeric:
            res[c] = self.rng.standard_normal(size=n)
        return res

    def generate_ts(
        self,
        num_days: int = 750,
        date_column: str = 'date'
    ) -> Dict[str, list]:
        res = {
            date_column: [date(2000, 1, 1) + timedelta(days=i) for i in range(num_days)]
        }
        for c in self.categorical_names:
            res[c] = self.rng.choice(ALPHABET, size=num_days)
        for c in self.num_numeric:
            res[c] = self.rng.standard_normal(size=num_days)
        return res
