from pyspark.sql import SparkSession

from faas.config import ETLConfig, recommend
from faas.generate import GenerateSynthetic, convert_dict_to_list


def test_recommend(spark: SparkSession):
    d = GenerateSynthetic(num_categorical=2, num_numeric=2)
    df = spark.createDataFrame(data=convert_dict_to_list(d.generate_iid()))
    actual, messages = recommend(df=df, target_column='numeric_0')
    expected = ETLConfig(
        x_categorical_columns=['categorical_0', 'categorical_1'],
        x_numeric_features=['numeric_1'],
        target_column='numeric_0',
    )
    assert actual == expected, ' '.join(messages)
