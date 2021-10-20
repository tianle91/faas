import pandas as pd

from faas.scaler import StandardScalerSpark, get_mean_std
from tests.conftest import create_test_df


def test_get_mean_std(spark):
    df = create_test_df(spark)
    vals = get_mean_std(df, 'q')
    assert list(vals.keys()) == ['all'], vals
    mean, stddev = vals['all']
    assert mean > 0 and stddev > 0, str((mean, stddev))


def test_StandardScalerSpark(spark):
    df = create_test_df(spark)
    expected = df.toPandas()

    sc = StandardScalerSpark(column='q').fit(df)
    sc.transform(df).collect()

    actual = sc.inverse_transform(sc.transform(df)).toPandas()
    pd.testing.assert_frame_equal(left=expected, right=actual)

    sc = StandardScalerSpark(column='q', group_column='p')
    sc.fit(df)

    actual = sc.inverse_transform(sc.transform(df)).toPandas()
    pd.testing.assert_frame_equal(left=expected, right=actual)
