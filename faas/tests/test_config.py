from pyspark.sql import SparkSession

from faas.config import Config, FeatureConfig, TargetConfig, recommend_config
from faas.generate import GenerateSynthetic, convert_dict_to_list


def test_recommend_config(spark: SparkSession):
    d = GenerateSynthetic(num_categorical=2, num_numeric=2)
    df = spark.createDataFrame(data=convert_dict_to_list(d.generate_iid()))
    actual = recommend_config(df=df, target_column='numeric_0')
    expected = Config(
        feature=FeatureConfig(
            categorical_columns=['categorical_0', 'categorical_1'],
            numeric_columns=['numeric_1']
        ),
        target=TargetConfig(column='numeric_0')
    )
    assert actual == expected
