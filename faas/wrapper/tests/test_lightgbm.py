from pyspark.sql import SparkSession

from faas.generate import GenerateSynthetic, convert_dict_to_list
from faas.wrapper import ETLConfig
from faas.wrapper.lightgbm import ETLWrapperForLGBM


def test_ETLWrapperForLGBM(spark: SparkSession):
    conf = ETLConfig(
        target_column='numeric_0',
        x_categorical_columns=['categorical_0'],
        x_numeric_features=['numeric_1'],
    )
    ewlgbm = ETLWrapperForLGBM(conf)
    d = GenerateSynthetic(num_categorical=2, num_numeric=2)
    df = spark.createDataFrame(data=convert_dict_to_list(d.generate_iid()))
    ewlgbm.fit(df)
    ewlgbm.predict(df)