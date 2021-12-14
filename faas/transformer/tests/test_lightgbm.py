import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import StringType

from faas.generate import GenerateSynthetic, convert_dict_to_list
from faas.transformer.etl import ETLConfig, FeatureConfig, TargetConfig
from faas.transformer.lightgbm import ETLWrapperForLGBM


@pytest.mark.parametrize(
    ('num_categorical'),
    [
        pytest.param(2, id='some categorical features'),
        pytest.param(0, id='no categorical features'),
    ]
)
def test_LGBMWrapper_numeric_iid(spark: SparkSession, num_categorical: int):
    d = GenerateSynthetic(num_categorical=num_categorical, num_numeric=2)
    dict_of_lists = d.generate_iid()
    conf = ETLConfig(
        target=TargetConfig(column=d.numeric_names[0]),
        feature=FeatureConfig(
            categorical_columns=d.categorical_names,
            numeric_columns=d.numeric_names[1:],
        )
    )
    ewlgbm = ETLWrapperForLGBM(conf)
    df = spark.createDataFrame(data=convert_dict_to_list(dict_of_lists))
    ewlgbm.fit(df)
    ewlgbm.predict(df)


def test_LGBMWrapper_categorical_iid(spark: SparkSession, num_categorical: int = 2):
    d = GenerateSynthetic(num_categorical=num_categorical, num_numeric=2)
    dict_of_lists = d.generate_iid()
    conf = ETLConfig(
        target=TargetConfig(
            column=d.categorical_names[0],
            is_categorical=True
        ),
        feature=FeatureConfig(
            categorical_columns=d.categorical_names,
            numeric_columns=d.numeric_names[1:],
        )
    )
    ewlgbm = ETLWrapperForLGBM(conf)
    df = spark.createDataFrame(data=convert_dict_to_list(dict_of_lists))
    ewlgbm.fit(df)
    df_predict = ewlgbm.predict(df)
    assert isinstance(df_predict.schema[d.categorical_names[0]].dataType, StringType)


def test_LGBMWrapper_ts(spark: SparkSession):
    d = GenerateSynthetic(num_categorical=2, num_numeric=2)
    dict_of_lists = d.generate_ts(date_column='dt')
    conf = ETLConfig(
        target=TargetConfig(column=d.numeric_names[0]),
        feature=FeatureConfig(
            categorical_columns=d.categorical_names,
            numeric_columns=d.numeric_names[1:],
            date_column='dt',
        )
    )
    ewlgbm = ETLWrapperForLGBM(conf)
    df = spark.createDataFrame(data=convert_dict_to_list(dict_of_lists))
    ewlgbm.fit(df)
    ewlgbm.predict(df)
