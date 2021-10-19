from faas.encoder import OrdinalEncoderSingleSpark
from tests.conftest import create_test_df


def test_OrdinalEncoderSingleSpark(spark):
    df = create_test_df(spark)
    enc = OrdinalEncoderSingleSpark('p')
    enc.fit(df)
    new_df = enc.transform(df)
    encoded_values = {row.p for row in new_df.select('p').distinct().collect()}
    assert encoded_values == {i for i in range(26)}
