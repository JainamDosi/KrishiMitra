# Databricks notebook source
# MAGIC %md
# MAGIC ## KrishiMitra — Delta Lake ETL Pipeline (Silver/Gold Layers)
# MAGIC
# MAGIC Transforms raw bronze data into enriched Delta Lake tables with PySpark:
# MAGIC 1. `mandi_prices` — Cleaned prices with window functions (lag, moving averages, volatility)
# MAGIC 2. Logging tables for disease predictions, price predictions, and chat sessions
# MAGIC
# MAGIC **Demonstrates:**
# MAGIC - Column renaming & type casting
# MAGIC - Date extraction (year, month, day_of_week)
# MAGIC - Window functions (lag, moving averages, std dev)
# MAGIC - Deduplication & null handling
# MAGIC - Delta Lake write with CDF enabled
# MAGIC - PK constraints & column comments
# MAGIC
# MAGIC **Configuration**: Use the widget inputs at the top of this notebook to set your catalog and schema.

# COMMAND ----------
# DBTITLE 1,Configuration & Setup

# Widget setup — configure your catalog and schema
dbutils.widgets.text("catalog", "krishimitra", "Catalog Name")
dbutils.widgets.text("schema", "agri_advisory", "Schema Name")

# Get configuration from widgets
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

spark.sql(f"USE CATALOG {catalog}")
spark.sql(f"USE SCHEMA {schema}")

print(f"Target: {catalog}.{schema}")

# COMMAND ----------
# DBTITLE 1,Load Raw Mandi Prices from Bronze

from pyspark.sql.functions import *
from pyspark.sql.window import Window
from pyspark.sql.types import *

# Read from bronze layer (created by notebook 01)
raw_df = spark.table(f"{catalog}.{schema}.mandi_prices_raw")
print(f"Raw records: {raw_df.count():,}")
print(f"Columns: {raw_df.columns}")

# COMMAND ----------
# DBTITLE 1,Clean & Transform — Parse Dates, Cast Types, Deduplicate

cleaned_df = (raw_df
    # Extract time features
    .withColumn("year", year("arrival_date"))
    .withColumn("month", month("arrival_date"))
    .withColumn("day_of_week", dayofweek("arrival_date"))
    # Filter valid rows
    .filter(col("modal_price").isNotNull() & (col("modal_price") > 0))
    .filter(col("arrival_date").isNotNull())
    # Deduplicate
    .dropDuplicates(["arrival_date", "market", "commodity", "variety"])
)

print(f"Cleaned records: {cleaned_df.count():,}")
cleaned_df.printSchema()

# COMMAND ----------
# DBTITLE 1,Spark Window Functions — Price Analytics

# This demonstrates advanced PySpark window operations for the hackathon
price_window = Window.partitionBy("commodity", "market").orderBy("arrival_date")

enriched_df = (cleaned_df
    # Previous day's price (lag)
    .withColumn("prev_price", lag("modal_price", 1).over(price_window))

    # Daily % change
    .withColumn("price_change_pct",
        when(col("prev_price").isNotNull(),
            round(((col("modal_price") - col("prev_price")) / col("prev_price") * 100), 2)
        ).otherwise(0.0))

    # 7-day moving average
    .withColumn("moving_avg_7d",
        round(avg("modal_price").over(price_window.rowsBetween(-6, 0)), 2))

    # 30-day moving average
    .withColumn("moving_avg_30d",
        round(avg("modal_price").over(price_window.rowsBetween(-29, 0)), 2))

    # Price volatility (std dev over 7 days)
    .withColumn("volatility_7d",
        round(stddev("modal_price").over(price_window.rowsBetween(-6, 0)), 2))

    # Drop temp column
    .drop("prev_price")
)

print(f"Enriched records: {enriched_df.count():,}")
enriched_df.show(5)

# COMMAND ----------
# DBTITLE 1,Write Enriched Prices to Delta Lake

enriched_df.write.format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(f"{catalog}.{schema}.mandi_prices")

print(f"Written to Delta: {catalog}.{schema}.mandi_prices")

# COMMAND ----------
# DBTITLE 1,Add PK & Enable CDF on mandi_prices

# Primary key
spark.sql(f"ALTER TABLE {catalog}.{schema}.mandi_prices ALTER COLUMN price_id SET NOT NULL")
spark.sql(f"""
  ALTER TABLE {catalog}.{schema}.mandi_prices
  ADD CONSTRAINT mandi_prices_enriched_pk PRIMARY KEY (price_id)
""")

# Change Data Feed
spark.sql(f"""ALTER TABLE {catalog}.{schema}.mandi_prices
             SET TBLPROPERTIES (delta.enableChangeDataFeed = true)""")

print("PK constraint and CDF enabled on mandi_prices")

# COMMAND ----------
# DBTITLE 1,Add Table & Column Comments on mandi_prices

spark.sql(f"""
  COMMENT ON TABLE {catalog}.{schema}.mandi_prices IS
  'Enriched commodity mandi prices with time features and technical indicators. Contains cleaned, deduplicated records with 7-day and 30-day moving averages, daily price change percentages, and volatility metrics. Derived from mandi_prices_raw via PySpark window functions. Primary key: price_id.'
""")

mandi_column_comments = {
    "price_id":          "Unique auto-generated UUID for each price record. Primary key.",
    "state":             "Indian state where the mandi is located.",
    "district":          "District within the state.",
    "market":            "Name of the agricultural mandi/market.",
    "commodity":         "Name of the traded commodity.",
    "variety":           "Variety of the commodity (e.g. Dara, Local, Hybrid).",
    "grade":             "Quality grade (e.g. FAQ - Fair Average Quality).",
    "arrival_date":      "Date when the price was recorded at the mandi.",
    "min_price":         "Minimum wholesale price per quintal in INR.",
    "max_price":         "Maximum wholesale price per quintal in INR.",
    "modal_price":       "Modal wholesale price per quintal in INR. Primary price indicator.",
    "source_file":       "Auto Loader metadata: source file path for lineage tracking.",
    "year":              "Year extracted from arrival_date for partitioning and time-series analysis.",
    "month":             "Month (1-12) extracted from arrival_date for seasonal analysis.",
    "day_of_week":       "Day of week (1=Sunday to 7=Saturday) from arrival_date. Used for weekly pattern detection.",
    "price_change_pct":  "Daily percentage change in modal_price compared to previous trading day. Partitioned by commodity+market.",
    "moving_avg_7d":     "7-day simple moving average of modal_price. Smooths short-term fluctuations.",
    "moving_avg_30d":    "30-day simple moving average of modal_price. Shows medium-term trend.",
    "volatility_7d":     "Standard deviation of modal_price over trailing 7 days. Higher values indicate price instability.",
}

for col_name, comment in mandi_column_comments.items():
    safe = comment.replace("'", "\\'")
    spark.sql(f"ALTER TABLE {catalog}.{schema}.mandi_prices ALTER COLUMN {col_name} COMMENT '{safe}'")

print("Table and column comments added for mandi_prices")

# COMMAND ----------
# DBTITLE 1,Price Summary Statistics

print("Price Summary by Commodity:")
spark.sql(f"""
    SELECT commodity,
           COUNT(*) as records,
           ROUND(AVG(modal_price), 2) as avg_price,
           ROUND(MIN(modal_price), 2) as min_price,
           ROUND(MAX(modal_price), 2) as max_price,
           MIN(arrival_date) as earliest,
           MAX(arrival_date) as latest
    FROM {catalog}.{schema}.mandi_prices
    GROUP BY commodity
    ORDER BY records DESC
    LIMIT 20
""").show(truncate=False)

# COMMAND ----------
# DBTITLE 1,Create Logging Tables

# Disease Predictions Log
spark.sql(f"""
CREATE TABLE IF NOT EXISTS {catalog}.{schema}.disease_predictions_log (
    prediction_id STRING,
    timestamp TIMESTAMP,
    predicted_disease STRING,
    predicted_crop STRING,
    confidence DOUBLE,
    treatment STRING,
    model_version STRING,
    user_language STRING
) USING DELTA
""")

spark.sql(f"""
  COMMENT ON TABLE {catalog}.{schema}.disease_predictions_log IS
  'Log of crop disease predictions from the HuggingFace MobileNetV2 classifier. Each row is one user-submitted image classification with predicted disease, confidence score, and treatment recommendation.'
""")

# Enable CDF
spark.sql(f"ALTER TABLE {catalog}.{schema}.disease_predictions_log SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")
print(f"Created: {catalog}.{schema}.disease_predictions_log")

# COMMAND ----------
# DBTITLE 1,Create Price Predictions Log Table

spark.sql(f"""
CREATE TABLE IF NOT EXISTS {catalog}.{schema}.price_predictions_log (
    prediction_id STRING,
    timestamp TIMESTAMP,
    commodity STRING,
    market STRING,
    state STRING,
    predicted_price DOUBLE,
    days_ahead INT,
    model_version STRING
) USING DELTA
""")

spark.sql(f"""
  COMMENT ON TABLE {catalog}.{schema}.price_predictions_log IS
  'Log of commodity price predictions from the Spark MLlib GBTRegressor model. Each row is one prediction request with commodity, market, predicted price, and forecast horizon.'
""")

spark.sql(f"ALTER TABLE {catalog}.{schema}.price_predictions_log SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")
print(f"Created: {catalog}.{schema}.price_predictions_log")

# COMMAND ----------
# DBTITLE 1,Create Chat Sessions Log Table

spark.sql(f"""
CREATE TABLE IF NOT EXISTS {catalog}.{schema}.chat_sessions (
    session_id STRING,
    timestamp TIMESTAMP,
    user_query STRING,
    intent STRING,
    response_summary STRING,
    language STRING,
    feature_used STRING
) USING DELTA
""")

spark.sql(f"""
  COMMENT ON TABLE {catalog}.{schema}.chat_sessions IS
  'Log of user chat interactions with KrishiMitra. Tracks queries, detected intents (disease_detection, price_prediction, scheme_advisory, pesticide_advisory), languages used, and features invoked.'
""")

spark.sql(f"ALTER TABLE {catalog}.{schema}.chat_sessions SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")
print(f"Created: {catalog}.{schema}.chat_sessions")

# COMMAND ----------
# DBTITLE 1,Verify All Delta Tables

print("=" * 60)
print("DELTA LAKE ETL SUMMARY")
print("=" * 60)
tables = [
    f"{catalog}.{schema}.mandi_prices_raw",
    f"{catalog}.{schema}.mandi_prices",
    f"{catalog}.{schema}.govt_schemes",
    f"{catalog}.{schema}.crop_knowledge",
    f"{catalog}.{schema}.pesticide_fertilizer_guide",
    f"{catalog}.{schema}.disease_predictions_log",
    f"{catalog}.{schema}.price_predictions_log",
    f"{catalog}.{schema}.chat_sessions",
]

for tname in tables:
    try:
        count = spark.table(tname).count()
        print(f"  {tname}: {count:,} rows")
    except Exception as e:
        print(f"  {tname}: (empty or error: {str(e)[:50]})")
print("=" * 60)
