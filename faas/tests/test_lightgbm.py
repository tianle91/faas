import pytest
from pyspark.sql import SparkSession

from faas.config import Config, FeatureConfig, TargetConfig
from faas.generate import GenerateSynthetic, convert_dict_to_list
from faas.lightgbm import LGBMWrapper


@pytest.mark.parametrize(
    ('num_categorical'),
    [
        pytest.param(2, id='some categorical'),
        pytest.param(0, id='no categorical'),
    ]
)
def test_LGBMWrapper_iid(spark: SparkSession, num_categorical: int):
    d = GenerateSynthetic(num_categorical=num_categorical, num_numeric=2)
    dict_of_lists = d.generate_iid()
    conf = Config(
        target=TargetConfig(column=d.numeric_names[0]),
        feature=FeatureConfig(
            categorical_columns=d.categorical_names,
            numeric_columns=d.numeric_names[1:],
        )
    )
    ewlgbm = LGBMWrapper(conf)
    df = spark.createDataFrame(data=convert_dict_to_list(dict_of_lists))
    ewlgbm.fit(df)
    ewlgbm.predict(df)


def test_LGBMWrapper_ts(spark: SparkSession):
    d = GenerateSynthetic(num_categorical=2, num_numeric=2)
    dict_of_lists = d.generate_ts(date_column='dt')
    conf = Config(
        target=TargetConfig(column=d.numeric_names[0]),
        feature=FeatureConfig(
            categorical_columns=d.categorical_names,
            numeric_columns=d.numeric_names[1:],
            date_column='dt',
        )
    )
    ewlgbm = LGBMWrapper(conf)
    df = spark.createDataFrame(data=convert_dict_to_list(dict_of_lists))
    ewlgbm.fit(df)
    ewlgbm.predict(df)
