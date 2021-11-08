import pandas as pd
import pytest
from pyspark.sql import DataFrame, SparkSession


@pytest.fixture(scope='session')
def spark():
    return SparkSession.builder.getOrCreate()
