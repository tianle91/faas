from faas.scaler import get_mean_std
from tests.conftest import create_test_df


def test_get_mean_std(spark):
    df = create_test_df(spark)
    vals = get_mean_std(df, 'q')
    assert list(vals.keys()) == ['all'], vals
    mean, stddev = vals['all']
    assert mean > 0 and stddev > 0, str((mean, stddev))
