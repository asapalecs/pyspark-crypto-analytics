from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType, StringType, StructField, StructType
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, RegressionEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression

DATA_PATH = "data/coin_USDCoin.csv"

FEATURES = [
    "prev_close", "prev_open", "prev_high", "prev_low",
    "prev_volume", "prev_range", "ma_7", "ma_30",
    "volume_ma_7", "prev_return",
]

SCHEMA = StructType([
    StructField("SNo", IntegerType()),
    StructField("Name", StringType()),
    StructField("Symbol", StringType()),
    StructField("Date", StringType()),
    StructField("High", DoubleType()),
    StructField("Low", DoubleType()),
    StructField("Open", DoubleType()),
    StructField("Close", DoubleType()),
    StructField("Volume", DoubleType()),
    StructField("Marketcap", DoubleType()),
])


def header(title):
    print(f"\n{'=' * 50}\n{title}\n{'=' * 50}")


def build_session():
    return (
        SparkSession.builder
        .appName("USDC Analysis")
        .config("spark.sql.adaptive.enabled", "true")
        .getOrCreate()
    )


def load(spark, path):
    df = spark.read.csv(path, header=True, schema=SCHEMA)
    return df.withColumn("Date", F.to_date("Date", "yyyy-MM-dd HH:mm:ss"))


def quality_report(df):
    header("CALITATEA DATELOR")
    total = df.count()
    nulls = df.select([F.sum(F.col(c).isNull().cast("int")).alias(c) for c in df.columns]).first().asDict()
    violations = df.filter(
        (F.col("High") < F.col("Low"))
        | (F.col("High") < F.col("Open"))
        | (F.col("High") < F.col("Close"))
        | (F.col("Low") > F.col("Open"))
        | (F.col("Low") > F.col("Close"))
    ).count()
    duplicates = total - df.select("Date").distinct().count()
    first, last = df.select(F.min("Date"), F.max("Date")).first()

    print(f"Randuri: {total}")
    print("Valori nule:", {k: v for k, v in nulls.items() if v})
    print(f"Incalcari OHLC: {violations}")
    print(f"Date duplicate: {duplicates}")
    print(f"Interval: {first} - {last}")

    if violations or duplicates:
        raise ValueError("Verificarea calitatii datelor a esuat")


def monthly_aggregates(df):
    return (
        df.groupBy(F.year("Date").alias("year"), F.month("Date").alias("month"))
        .agg(
            F.avg("Close").alias("avg_close"),
            F.stddev("Close").alias("stddev_close"),
            F.sum("Volume").alias("total_volume"),
            F.count("*").alias("days"),
        )
        .orderBy("year", "month")
    )


def engineer_features(df):
    w = Window.orderBy("Date")
    out = (
        df.withColumn("prev_close", F.lag("Close", 1).over(w))
        .withColumn("prev_open", F.lag("Open", 1).over(w))
        .withColumn("prev_high", F.lag("High", 1).over(w))
        .withColumn("prev_low", F.lag("Low", 1).over(w))
        .withColumn("prev_volume", F.lag("Volume", 1).over(w))
        .withColumn("prev_range", F.col("prev_high") - F.col("prev_low"))
        # ferestrele se opresc la -1 ca ziua curenta sa nu intre in propriile features
        .withColumn("ma_7", F.avg("Close").over(w.rowsBetween(-7, -1)))
        .withColumn("ma_30", F.avg("Close").over(w.rowsBetween(-30, -1)))
        .withColumn("volume_ma_7", F.avg("Volume").over(w.rowsBetween(-7, -1)))
        .withColumn("prev_return", F.col("prev_close") / F.lag("Close", 2).over(w) - 1)
    )
    return out.filter(F.col("ma_30").isNotNull())


def chronological_split(df, train_fraction=0.8, verbose=True):
    cutoff_row = int(df.count() * train_fraction)
    cutoff = df.orderBy("Date").limit(cutoff_row).agg(F.max("Date")).first()[0]
    train = df.filter(F.col("Date") <= cutoff)
    test = df.filter(F.col("Date") > cutoff)
    if verbose:
        print(f"\nImpartire la {cutoff}: antrenare={train.count()} test={test.count()}")
    return train, test


def run_regression(train, test):
    header("REGRESIE: pretul de inchidere de maine")

    evaluator = RegressionEvaluator(labelCol="Close", metricName="rmse")
    fit_part, val_part = chronological_split(train, verbose=False)

    def pipeline(reg, mix):
        assembler = VectorAssembler(inputCols=FEATURES, outputCol="features", handleInvalid="skip")
        lr = LinearRegression(featuresCol="features", labelCol="Close", regParam=reg, elasticNetParam=mix)
        return Pipeline(stages=[assembler, lr])

    # tuning pe finalul perioadei de antrenare, nu pe randuri amestecate
    scores = {
        (reg, mix): evaluator.evaluate(pipeline(reg, mix).fit(fit_part).transform(val_part))
        for reg in [0.0, 0.001, 0.01, 0.1, 1.0]
        for mix in [0.0, 0.5, 1.0]
    }
    reg, mix = min(scores, key=scores.get)
    model = pipeline(reg, mix).fit(train)

    train_rmse = evaluator.evaluate(model.transform(train))
    test_rmse = evaluator.evaluate(model.transform(test))
    baseline_rmse = test.select(
        F.sqrt(F.avg(F.pow(F.col("Close") - F.col("prev_close"), 2)))
    ).first()[0]

    print(f"Parametri optimi: regParam={reg}, elasticNetParam={mix}")
    print(f"RMSE antrenare:   {train_rmse:.6f}")
    print(f"RMSE test:        {test_rmse:.6f}")
    print(f"RMSE baseline:    {baseline_rmse:.6f} (maine = azi)")
    print(f"Bate baseline-ul: {'da' if test_rmse < baseline_rmse else 'nu'}")


def run_classification(train, test):
    header("CLASIFICARE: zi cu volatilitate mare")

    def with_label(df):
        return df.withColumn("volatile", F.when(F.col("High") - F.col("Low") > 0.01, 1).otherwise(0))

    train, test = with_label(train), with_label(test)
    balance = {r["volatile"]: r["count"] for r in train.groupBy("volatile").count().collect()}
    print("Distributia claselor (antrenare):", balance)

    assembler = VectorAssembler(inputCols=FEATURES, outputCol="features", handleInvalid="skip")
    clf = LogisticRegression(featuresCol="features", labelCol="volatile")
    model = Pipeline(stages=[assembler, clf]).fit(train)

    auc = BinaryClassificationEvaluator(labelCol="volatile").evaluate(model.transform(test))
    print(f"AUC test: {auc:.4f}")


def categorise_volume(df):
    return df.withColumn(
        "volume_category",
        F.when(F.col("Volume") < 1e6, "low")
        .when(F.col("Volume") < 10e6, "medium")
        .otherwise("high"),
    )


def main():
    spark = build_session()
    spark.sparkContext.setLogLevel("ERROR")

    try:
        df = load(spark, DATA_PATH)
        quality_report(df)

        header("AGREGARI LUNARE")
        monthly_aggregates(df).show(12, truncate=False)

        df.createOrReplaceTempView("usdc")
        spark.sql("""
            SELECT year(Date) AS year, month(Date) AS month,
                   AVG(Close) AS avg_close, SUM(Volume) AS total_volume
            FROM usdc
            GROUP BY year(Date), month(Date)
            ORDER BY year, month
        """).show(12)

        featured = engineer_features(df).cache()
        train, test = chronological_split(featured)
        run_regression(train, test)
        run_classification(train, test)

        header("CATEGORII DE VOLUM")
        categorise_volume(df).groupBy("volume_category").count().show()

        featured.unpersist()
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
