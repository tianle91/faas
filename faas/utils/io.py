import os
from io import BytesIO
from tempfile import TemporaryDirectory

from pyspark.sql import DataFrame, SparkSession
from streamlit.uploaded_file_manager import UploadedFile


def dump_file_to_location(file: BytesIO, p: str):
    with open(p, 'wb') as f:
        f.write(file.read())


EXT_TO_SPARK_READ_OPTIONS = {
    '.csv': ('csv', {'header': True, 'inferSchema': True}),
    '.parquet': ('parquet', {}),
}


def load_from_path(spark: SparkSession, p: str) -> DataFrame:
    _, ext = os.path.splitext(p)
    format_name, options = EXT_TO_SPARK_READ_OPTIONS[ext]
    return spark.read.format(format_name).options(**options).load(p)


def load_cached_df_from_st_uploaded(f: UploadedFile, spark: SparkSession) -> DataFrame:
    with TemporaryDirectory() as temp_dir:
        p = os.path.join(temp_dir, f.name)
        dump_file_to_location(f, p=p)
        df = load_from_path(spark=spark, p=p)
        # cache df in memory
        df.cache()
        df.count()
    return df
