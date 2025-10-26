# Databricks + Spark ML — Exemplo mínimo

# > Este notebook funciona no Databricks (ou local com PySpark). Demonstra leitura de dados, transformação e um pipeline de ML.


from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline

spark = SparkSession.builder.appName("databricks-demo").getOrCreate()

# Exemplo: Iris CSV disponível em DBFS ou upload
# df = spark.read.csv("/databricks-datasets/iris/iris.csv", header=True, inferSchema=True)
data = [(5.1,3.5,1.4,0.2,"setosa"),
        (6.7,3.1,4.7,1.5,"versicolor"),
        (6.5,3.0,5.8,2.2,"virginica")]
df = spark.createDataFrame(data, ["sepal_length","sepal_width","petal_length","petal_width","label"])

idx = StringIndexer(inputCol="label", outputCol="indexedLabel")
vec = VectorAssembler(inputCols=["sepal_length","sepal_width","petal_length","petal_width"], outputCol="features")
rf  = RandomForestClassifier(labelCol="indexedLabel", featuresCol="features", numTrees=50, seed=42)

pipeline = Pipeline(stages=[idx, vec, rf])
model = pipeline.fit(df)
pred  = model.transform(df)
pred.select("features","indexedLabel","prediction","probability").show(truncate=False)
