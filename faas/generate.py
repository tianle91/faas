from datetime import date, timedelta
from typing import Dict, List

from geopy import Point
from numpy.random import default_rng

ALPHABETS = [c for c in 'ABCDEFGHIJLKMNOPQRSTUVWXYZ']


def numeric(rng, n: int = 1) -> List[float]:
    return [float(v) for v in rng.standard_normal(size=n)]


def categorical(rng, n: int = 1) -> List[float]:
    return [str(v) for v in rng.choice(ALPHABETS, size=n)]


LOCATIONS = {
    'Toronto': Point(latitude=43.653908, longitude=-79.384293),
    'Tokyo': Point(latitude=35.652832, longitude=139.839478),
    'Sydney': Point(latitude=-33.870453, longitude=151.208755),
}


def location(rng, n: int = 1) -> List[Point]:
    return [v for v in rng.choice(list(LOCATIONS.values()), size=n)]


def convert_dict_to_list(d: Dict[str, list]) -> List[dict]:
    first_key = list(d.keys())[0]
    n = len(d[first_key])
    for k, v in d.items():
        if len(v) != n:
            raise ValueError(f'Value for key: {k} is not of length {n}! It is {len(v)} instead.')
    data = [
        {k: d[k][i] for k in d}
        for i in range(n)
    ]
    return data


class GenerateSynthetic:
    """Generate synthetic data.
    """

    def __init__(self, num_categorical: int = 2, num_numeric: int = 2):
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
            res[c] = categorical(self.rng, n)
        for c in self.numeric_names:
            res[c] = numeric(self.rng, n)
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
        res = {}
        for ts_type in ts_types:
            res_ts = self.generate_ts(num_days=num_days, date_column=date_column)
            res_ts[ts_type_column] = [ts_type, ] * num_days
            for k, l in res_ts.items():
                res[k] = res.get(k, []) + l
        return res

    def generate_spatial(
        self,
        num_locations: int = 5,
        latitude_column: str = 'lat',
        longitude_column: str = 'lon',
    ) -> Dict[str, list]:
        res = self.generate_iid(n=num_locations)
        locations = location(rng=self.rng, n=num_locations)
        lats, lons = [], []
        for p in locations:
            lats.append(p.latitude)
            lons.append(p.longitude)
        res[latitude_column] = lats
        res[longitude_column] = lons
        return res
