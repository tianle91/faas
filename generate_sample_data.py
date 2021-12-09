from pyspark.sql import DataFrame, SparkSession

from faas.generate import GenerateSynthetic, convert_dict_to_list

spark = SparkSession.builder.getOrCreate()
gs = GenerateSynthetic(num_categorical=2, num_numeric=2)

# sample iid
dict_of_lists = gs.generate_iid()
df: DataFrame = spark.createDataFrame(data=convert_dict_to_list(dict_of_lists))
df.toPandas().to_csv('data/sample_iid.csv', index=False)

# sample ts
dict_of_lists = gs.generate_ts()
df: DataFrame = spark.createDataFrame(data=convert_dict_to_list(dict_of_lists))
df.toPandas().to_csv('data/sample_ts.csv', index=False)

# sample ts parquet
dict_of_lists = gs.generate_ts()
df: DataFrame = spark.createDataFrame(data=convert_dict_to_list(dict_of_lists))
df.toPandas().to_parquet('data/sample_ts.parquet', index=False)

# sample multi ts
dict_of_lists = gs.generate_multi_ts(ts_types=['ts_A', 'ts_B'])
df: DataFrame = spark.createDataFrame(data=convert_dict_to_list(dict_of_lists))
df.toPandas().to_csv('data/sample_multi_ts.csv', index=False)

# sample spatial
dict_of_lists = gs.generate_spatial()
df: DataFrame = spark.createDataFrame(data=convert_dict_to_list(dict_of_lists))
df.toPandas().to_csv('data/sample_spatial.csv', index=False)

# sample numeric iid
gs_numeric = GenerateSynthetic(num_categorical=0, num_numeric=10)

dict_of_lists = gs_numeric.generate_iid()
df: DataFrame = spark.createDataFrame(data=convert_dict_to_list(dict_of_lists))
df.toPandas().to_csv('data/sample_numeric_iid.csv', index=False)

# sample categorical iid
gs_categorical = GenerateSynthetic(num_categorical=10, num_numeric=0)

dict_of_lists = gs_categorical.generate_iid()
df: DataFrame = spark.createDataFrame(data=convert_dict_to_list(dict_of_lists))
df.toPandas().to_csv('data/sample_categorical_iid.csv', index=False)
