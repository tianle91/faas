import pandas as pd

from faas.data import Preprocess
from tests.conftest import create_test_df


def test_Preprocess(spark):
    df = create_test_df(spark)
    expected = df.toPandas()

    pp = Preprocess('p').fit(df)
    pp.transform(df)

    actual = pp.inverse_transform(pp.transform(df)).toPandas()
    pd.testing.assert_frame_equal(left=expected, right=actual)
