import pandas as pd

from faas.generate import GenerateSynthetic


def test_iid():
    df = pd.DataFrame(GenerateSynthetic().generate_iid(n=100))
    assert len(df) == 100


def test_ts():
    df = pd.DataFrame(GenerateSynthetic().generate_ts(num_days=100))
    assert len(df) == 100


def test_multi_ts():
    df = pd.DataFrame(GenerateSynthetic().generate_multi_ts(ts_types=['a', 'b'], num_days=100))
    assert len(df) == 200


def test_spatial():
    df = pd.DataFrame(GenerateSynthetic().generate_spatial(num_locations=100))
    assert len(df) == 100
