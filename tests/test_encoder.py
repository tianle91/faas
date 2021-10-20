import pandas as pd

from faas.encoder import OrdinalEncoderSingleSpark
from tests.conftest import create_test_df


def test_OrdinalEncoderSingleSpark(spark):
    df = create_test_df(spark)
    expected = df.toPandas()

    enc = OrdinalEncoderSingleSpark('p').fit(df)
    encoded_rows = enc.transform(df).distinct().collect()

    encoded_values = {row.p for row in encoded_rows}
    assert encoded_values == {i for i in range(26)}

    actual = enc.inverse_transform(enc.transform(df)).toPandas()
    pd.testing.assert_frame_equal(left=expected, right=actual)
