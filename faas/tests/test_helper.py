from pyspark.sql import SparkSession

from faas.config import Config
import pytest
from faas.helper import get_trained, get_prediction


@pytest.mark.parametrize(
    ('p', 'conf'),
    [
        pytest.param(
            'data/sample_multi_ts.csv',
            # categorical_0,categorical_1,date,numeric_0,numeric_1,ts_type
            Config(
                target='numeric_0',
                date_column='date',
                group_columns=['ts_type'],
                feature_columns=['categorical_0', 'categorical_1', 'numeric_1'],
            ),
            id='sample_multi_ts'
        ),
    ]
)
def test_helpers(p: str, conf: Config):
    spark = SparkSession.builder.appName('test_helpers').getOrCreate()
    df = spark.read.options(header=True, inferSchema=True).csv(p)
    m = get_trained(conf=conf, df=df)
    df_pred, msgs = get_prediction(conf=conf, df=df.drop(conf.target), m=m)
    df_pred.toPandas()
    print('\n'.join(msgs))
