import os
import sys

# Evita PYTHON_VERSION_MISMATCH la UDF cand python3 din PATH are alta versiune
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import StringType
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import RegressionEvaluator, BinaryClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

spark = SparkSession.builder \
    .appName("USDCoin Analysis") \
    .getOrCreate()

file_path = "data/coin_USDCoin.csv"
df = spark.read.csv(file_path, header=True, inferSchema=True)
print("PAS 1: Initializez SparkSession si incarc Fisierul CSV.")
print(f"Numar inregistrari: {df.count()}, Coloane: {df.columns}")

df = df.withColumn("Date", to_date(col("Date"), "yyyy-MM-dd"))
df = df.dropna()
print("PAS 2: Curatam data.")
df.show()
df.printSchema()

monthly_stats = df.withColumn("Year", year("Date")) \
    .withColumn("Month", month("Date")) \
    .groupBy("Year", "Month") \
    .agg(
        avg("Close").alias("Average_Close"),
        sum("Volume").alias("Total_Volume")
    ).orderBy("Year", "Month")

monthly_stats.show()
print("PAS 3: Agregare lunara realizata cu DataFrame API.")

df.createOrReplaceTempView("usdc")
sql_result = spark.sql("""
    SELECT year(Date) as Year, month(Date) as Month,
           AVG(Close) as Avg_Close,
           SUM(Volume) as Total_Volume
    FROM usdc
    GROUP BY year(Date), month(Date)
    ORDER BY year(Date), month(Date)
""")
sql_result.show()
print("PAS 4: Agregare lunara cu Spark SQL.")

features = ["Open", "High", "Low", "Volume", "Marketcap"]
assembler = VectorAssembler(inputCols=features, outputCol="features")
df_ml = assembler.transform(df).select("features", "Close")
print("PAS 5: VectorAssembler a fost aplicat.")
df_ml.select("features", "Close").show(3)

lr = LinearRegression(featuresCol="features", labelCol="Close")
lr_model = lr.fit(df_ml)
predictions = lr_model.transform(df_ml)

reg_eval = RegressionEvaluator(labelCol="Close", predictionCol="prediction", metricName="rmse")
rmse = reg_eval.evaluate(predictions)
print(f"PAS 6: Model de regresie antrenat. RMSE: {rmse:.4f}")

df_class = df.withColumn("volatility", when((col("High") - col("Low")) > 0.01, 1).otherwise(0))
assembler_cls = VectorAssembler(inputCols=features, outputCol="features")
df_class = assembler_cls.transform(df_class).select("features", "volatility")
print("PAS 7: Eticheta 'volatility' a fost adaugata.")
df_class.select("features", "volatility").show()

lr_cls = LogisticRegression(featuresCol="features", labelCol="volatility")
model_cls = lr_cls.fit(df_class)
preds = model_cls.transform(df_class)

cls_eval = BinaryClassificationEvaluator(labelCol="volatility")
auc = cls_eval.evaluate(preds)
print(f"PAS 8: Model de clasificare antrenat. AUC: {auc:.4f}")

pipeline = Pipeline(stages=[assembler, lr])
model_pipeline = pipeline.fit(df)
pipeline_predictions = model_pipeline.transform(df)
pipeline_predictions.select("prediction", "Close").show()
print("PAS 9: Pipeline ML executat cu succes.")

def volume_category(volume):
    if volume is None:
        return None
    if volume < 1e6:
        return "low"
    elif volume < 10e6:
        return "medium"
    else:
        return "high"

volume_udf = udf(volume_category, StringType())
df = df.withColumn("VolumeCategory", volume_udf(col("Volume")))
print("PAS 10: UDF aplicat pentru categorizarea volumului.")
df.select("Volume", "VolumeCategory").show()

paramGrid = ParamGridBuilder() \
    .addGrid(lr.regParam, [0.01, 0.1, 1.0]) \
    .build()

crossval = CrossValidator(
    estimator=lr,
    estimatorParamMaps=paramGrid,
    evaluator=reg_eval,
    numFolds=3
)

cv_model = crossval.fit(df_ml)
best_model = cv_model.bestModel
best_rmse = reg_eval.evaluate(best_model.transform(df_ml))
print(f"PAS 11: CrossValidator a antrenat model: {best_rmse:.4f}")

spark.stop()
