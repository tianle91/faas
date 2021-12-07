import pytest
from pyspark.sql import SparkSession

from faas.config import Config
from faas.helper import get_prediction, get_trained

spark = SparkSession.builder.appName('test_helpers').getOrCreate()


@pytest.mark.parametrize(
    ('p', 'conf'),
    [
        pytest.param(
            'data/sample_iid.csv',
            # categorical_0,categorical_1,numeric_0,numeric_1
            Config(
                target='numeric_0',
                target_is_categorical=False,
                feature_columns=['categorical_0', 'categorical_1', 'numeric_1'],
            ),
            id='sample_iid:numeric'
        ),
        pytest.param(
            'data/sample_iid.csv',
            # categorical_0,categorical_1,numeric_0,numeric_1
            Config(
                target='categorical_0',
                target_is_categorical=True,
                feature_columns=['categorical_1', 'numeric_0', 'numeric_1'],
            ),
            id='sample_iid:categorical'
        ),
        # TODO: binary categorical
        pytest.param(
            'data/sample_ts.csv',
            # categorical_0,categorical_1,date,numeric_0,numeric_1
            Config(
                target='numeric_0',
                target_is_categorical=False,
                date_column='date',
                feature_columns=['categorical_0', 'categorical_1', 'numeric_1'],
            ),
            id='sample_ts:numeric'
        ),
        pytest.param(
            'data/sample_ts.csv',
            # categorical_0,categorical_1,date,numeric_0,numeric_1
            Config(
                target='categorical_0',
                target_is_categorical=True,
                date_column='date',
                feature_columns=['categorical_1', 'numeric_0', 'numeric_1'],
            ),
            id='sample_ts:categorical'
        ),
        pytest.param(
            'data/sample_multi_ts.csv',
            # categorical_0,categorical_1,date,numeric_0,numeric_1,ts_type
            Config(
                target='numeric_0',
                target_is_categorical=False,
                date_column='date',
                group_columns=['ts_type'],
                feature_columns=['categorical_0', 'categorical_1', 'numeric_1'],
            ),
            id='sample_multi_ts:numeric'
        ),
        pytest.param(
            'data/sample_multi_ts.csv',
            # categorical_0,categorical_1,date,numeric_0,numeric_1,ts_type
            Config(
                target='ts_type',
                target_is_categorical=True,
                date_column='date',
                feature_columns=['categorical_0', 'categorical_1', 'numeric_0', 'numeric_1'],
            ),
            id='sample_multi_ts:binary'
        ),
        pytest.param(
            'data/sample_spatial.csv',
            # categorical_0,categorical_1,lat,lon,numeric_0,numeric_1
            Config(
                target='numeric_0',
                target_is_categorical=False,
                latitude_column='lat',
                longitude_column='lon',
                feature_columns=['categorical_0', 'categorical_1', 'numeric_1'],
            ),
            id='sample_spatial:numeric'
        ),
        pytest.param(
            'data/sample_numeric_iid.csv',
            # numeric_0, ..., numeric_9
            Config(
                target='numeric_0',
                target_is_categorical=False,
                feature_columns=[f'numeric_{i}' for i in range(1, 10)],
            ),
            id='sample_numeric_iid'
        ),
        pytest.param(
            'data/sample_categorical_iid.csv',
            # categorical_0, ..., categorical_9
            Config(
                target='categorical_0',
                target_is_categorical=True,
                feature_columns=[f'categorical_{i}' for i in range(1, 10)],
            ),
            id='sample_categorical_iid'
        ),
    ]
)
def test_helpers(p: str, conf: Config):
    df = spark.read.options(header=True, inferSchema=True).csv(p)
    m = get_trained(conf=conf, df=df)
    df_pred, msgs = get_prediction(conf=conf, df=df.drop(conf.target), m=m)
    df_pred.toPandas()
    print('\n'.join(msgs))
