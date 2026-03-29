# Databricks notebook source
# MAGIC %md
# MAGIC ## KrishiMitra — Price Prediction Model (Spark MLlib)
# MAGIC
# MAGIC Trains a commodity price prediction model using:
# MAGIC - **StringIndexer** for categorical features (commodity, market, state)
# MAGIC - **VectorAssembler** to combine all features
# MAGIC - **StandardScaler** for normalization
# MAGIC - **GBTRegressor** (Gradient Boosted Trees) for prediction
# MAGIC - **MLflow** for experiment tracking and model registration
# MAGIC
# MAGIC **Configuration**: Use the widget inputs at the top of this notebook to set your catalog and schema.

# COMMAND ----------
# DBTITLE 1,Configuration & Setup

dbutils.widgets.text("catalog", "krishimitra", "Catalog Name")
dbutils.widgets.text("schema", "agri_advisory", "Schema Name")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

spark.sql(f"USE CATALOG {catalog}")
spark.sql(f"USE SCHEMA {schema}")

print(f"Target: {catalog}.{schema}")

# COMMAND ----------
# DBTITLE 1,Load Enriched Data from Delta Lake

from pyspark.sql.functions import *
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
import mlflow
import mlflow.spark

prices_df = spark.table(f"{catalog}.{schema}.mandi_prices") \
    .filter(col("moving_avg_7d").isNotNull()) \
    .filter(col("moving_avg_30d").isNotNull()) \
    .filter(col("modal_price") > 0)

print(f"Total records: {prices_df.count():,}")
print(f"Columns: {prices_df.columns}")

# Show data distribution
prices_df.groupBy("commodity").count().orderBy(desc("count")).show(20, truncate=False)

# COMMAND ----------
# DBTITLE 1,Feature Engineering Pipeline

# Categorical feature encoders
commodity_indexer = StringIndexer(
    inputCol="commodity", outputCol="commodity_idx", handleInvalid="keep"
)
market_indexer = StringIndexer(
    inputCol="market", outputCol="market_idx", handleInvalid="keep"
)
state_indexer = StringIndexer(
    inputCol="state", outputCol="state_idx", handleInvalid="keep"
)

# Feature columns
feature_cols = [
    "year", "month", "day_of_week",  # Time features
    "commodity_idx", "market_idx", "state_idx",  # Categorical (encoded)
    "moving_avg_7d", "moving_avg_30d",  # Technical indicators
]

# Vector assembler
assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="raw_features",
    handleInvalid="skip",
)

# Standard scaler
scaler = StandardScaler(
    inputCol="raw_features",
    outputCol="features",
    withStd=True,
    withMean=True,
)

print(f"Feature pipeline configured: {feature_cols}")

# COMMAND ----------
# DBTITLE 1,GBT Regressor Configuration

gbt = GBTRegressor(
    featuresCol="features",
    labelCol="modal_price",
    maxIter=50,
    maxDepth=6,
    stepSize=0.1,
    minInstancesPerNode=5,
    subsamplingRate=0.8,
    seed=42,
)

# Full ML Pipeline
pipeline = Pipeline(stages=[
    commodity_indexer,
    market_indexer,
    state_indexer,
    assembler,
    scaler,
    gbt,
])

print("GBTRegressor pipeline configured")
print(f"   - maxIter: 50")
print(f"   - maxDepth: 6")
print(f"   - stepSize: 0.1")
print(f"   - features: {len(feature_cols)}")

# COMMAND ----------
# DBTITLE 1,Train/Test Split & Model Training

train_df, test_df = prices_df.randomSplit([0.8, 0.2], seed=42)
print(f"Train: {train_df.count():,} | Test: {test_df.count():,}")

# COMMAND ----------
# DBTITLE 1,Train with MLflow Tracking

mlflow.set_experiment(f"/{catalog}/krishimitra/price-prediction")

with mlflow.start_run(run_name="gbt_price_v1"):
    # Train the pipeline
    print("Training GBTRegressor pipeline...")
    model = pipeline.fit(train_df)
    print("Training complete!")

    # Evaluate
    predictions = model.transform(test_df)

    evaluator_rmse = RegressionEvaluator(
        labelCol="modal_price", predictionCol="prediction", metricName="rmse"
    )
    evaluator_mae = RegressionEvaluator(
        labelCol="modal_price", predictionCol="prediction", metricName="mae"
    )
    evaluator_r2 = RegressionEvaluator(
        labelCol="modal_price", predictionCol="prediction", metricName="r2"
    )

    rmse = evaluator_rmse.evaluate(predictions)
    mae = evaluator_mae.evaluate(predictions)
    r2 = evaluator_r2.evaluate(predictions)

    print(f"\nEvaluation Results:")
    print(f"   RMSE:  {rmse:,.2f}")
    print(f"   MAE:   {mae:,.2f}")
    print(f"   R²:    {r2:.4f}")

    # Log parameters
    mlflow.log_params({
        "model": "GBTRegressor",
        "max_iter": 50,
        "max_depth": 6,
        "step_size": 0.1,
        "features": str(feature_cols),
        "num_features": len(feature_cols),
        "train_size": train_df.count(),
        "test_size": test_df.count(),
    })

    # Log metrics
    mlflow.log_metrics({
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    })

    # Register model
    mlflow.spark.log_model(
        model,
        "price-predictor",
        registered_model_name="krishimitra-price-predictor",
    )

    print(f"\nModel registered as 'krishimitra-price-predictor'")

# COMMAND ----------
# DBTITLE 1,Prediction Analysis — Sample Results

# Show sample predictions vs actuals
print("Sample Predictions vs Actuals:")
predictions.select(
    "commodity", "market", "arrival_date",
    "modal_price", "prediction"
).withColumn(
    "error", round(col("prediction") - col("modal_price"), 2)
).withColumn(
    "error_pct", round((col("prediction") - col("modal_price")) / col("modal_price") * 100, 2)
).show(20, truncate=False)

# COMMAND ----------
# DBTITLE 1,Error Distribution by Commodity

from pyspark.sql.functions import sqrt as spark_sqrt

print("RMSE by Commodity (Top 10):")
predictions.groupBy("commodity").agg(
    spark_sqrt(avg(pow(col("prediction") - col("modal_price"), 2))).alias("rmse"),
    avg(abs(col("prediction") - col("modal_price"))).alias("mae"),
    count("*").alias("samples"),
).orderBy("rmse").show(10, truncate=False)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Price Prediction Model Complete
# MAGIC
# MAGIC | Property | Value |
# MAGIC |----------|-------|
# MAGIC | Algorithm | GBTRegressor (Gradient Boosted Trees) |
# MAGIC | Framework | Spark MLlib Pipeline |
# MAGIC | Features | 8 (time + categorical + moving averages) |
# MAGIC | Registry | `models:/krishimitra-price-predictor/latest` |
# MAGIC | Tracking | MLflow experiment: `/{catalog}/krishimitra/price-prediction` |
