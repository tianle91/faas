import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType

from faas.generate import GenerateSynthetic, convert_dict_to_list


def test_iid(spark: SparkSession):
    d = GenerateSynthetic().generate_iid(n=100)
    assert len(pd.DataFrame(d)) == 100
    sdf = spark.createDataFrame(data=convert_dict_to_list(d))
    assert sdf.count() == 100


def test_ts():
    df = pd.DataFrame(GenerateSynthetic().generate_ts(num_days=100))
    assert len(df) == 100


def test_multi_ts():
    df = pd.DataFrame(GenerateSynthetic().generate_multi_ts(ts_types=['a', 'b'], num_days=100))
    assert len(df) == 200


def test_spatial():
    df = pd.DataFrame(GenerateSynthetic().generate_spatial(num_locations=100))
    assert len(df) == 100


def test_convert_dict_to_list():
    d = GenerateSynthetic().generate_iid(n=100)
    actual = convert_dict_to_list(d=d)
    assert len(actual) == 100
