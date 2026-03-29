"""
KrishiMitra — Delta Lake Read/Write Utilities
===============================================
Helpers for reading/writing Delta Lake tables, MERGE upserts,
and session logging. Works both in Databricks and local Spark.
"""

import logging
import uuid
from datetime import datetime
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Spark Session Management
# ──────────────────────────────────────────────

_spark = None


def get_spark():
    """
    Get or create a SparkSession. Uses existing Databricks session
    if available, otherwise creates a local session with Delta support.
    """
    global _spark
    if _spark is not None:
        return _spark

    try:
        # Try Databricks environment first
        from databricks.connect import DatabricksSession
        _spark = DatabricksSession.builder.getOrCreate()
        logger.info("✅ Using Databricks SparkSession")
    except ImportError:
        try:
            # Try existing spark session (Databricks notebook)
            from pyspark.sql import SparkSession
            _spark = SparkSession.builder.getOrCreate()
            if _spark.conf.get("spark.databricks.clusterUsageTags.clusterId", None):
                logger.info("✅ Using existing Databricks cluster SparkSession")
            else:
                raise Exception("Not on Databricks")
        except:
            # Local development — create with Delta
            from pyspark.sql import SparkSession
            _spark = (
                SparkSession.builder
                .appName("KrishiMitra-Local")
                .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
                .config(
                    "spark.sql.catalog.spark_catalog",
                    "org.apache.spark.sql.delta.catalog.DeltaCatalogExtension",
                )
                .config("spark.jars.packages", "io.delta:delta-spark_2.12:3.0.0")
                .config("spark.driver.memory", "4g")
                .master("local[*]")
                .getOrCreate()
            )
            logger.info("✅ Created local SparkSession with Delta Lake")

    return _spark


# ──────────────────────────────────────────────
# Delta Lake Table Operations
# ──────────────────────────────────────────────

def table_exists(table_name: str) -> bool:
    """Check if a Delta table exists."""
    spark = get_spark()
    try:
        spark.table(table_name)
        return True
    except Exception:
        return False


def read_table(table_name: str, filters: Optional[Dict[str, Any]] = None):
    """
    Read a Delta Lake table with optional filters.

    Args:
        table_name: Fully qualified table name (e.g., 'krishimitra.mandi_prices')
        filters: Dict of column → value filters

    Returns:
        PySpark DataFrame
    """
    spark = get_spark()
    df = spark.table(table_name)

    if filters:
        from pyspark.sql.functions import col
        for column, value in filters.items():
            if isinstance(value, list):
                df = df.filter(col(column).isin(value))
            else:
                df = df.filter(col(column) == value)

    return df


def read_table_as_pandas(table_name: str, filters: Optional[Dict[str, Any]] = None,
                          limit: Optional[int] = None):
    """Read a Delta table and return as Pandas DataFrame."""
    df = read_table(table_name, filters)
    if limit:
        df = df.limit(limit)
    return df.toPandas()


def write_table(df, table_name: str, mode: str = "overwrite",
                partition_by: Optional[List[str]] = None):
    """
    Write a DataFrame to a Delta Lake table.

    Args:
        df: PySpark or Pandas DataFrame
        table_name: Target table name
        mode: 'overwrite', 'append', 'merge'
        partition_by: Optional partition columns
    """
    spark = get_spark()

    # Convert Pandas to Spark if needed
    if hasattr(df, "to_spark"):
        df = df.to_spark()
    elif not hasattr(df, "write"):
        # Pandas DataFrame
        df = spark.createDataFrame(df)

    writer = df.write.format("delta").mode(mode).option("overwriteSchema", "true")

    if partition_by:
        writer = writer.partitionBy(*partition_by)

    writer.saveAsTable(table_name)
    logger.info(f"✅ Wrote {df.count()} rows to {table_name} (mode={mode})")


def execute_sql(query: str):
    """Execute a SparkSQL query and return the result DataFrame."""
    spark = get_spark()
    return spark.sql(query)


# ──────────────────────────────────────────────
# Prediction Logging
# ──────────────────────────────────────────────

def log_disease_prediction(prediction: Dict[str, Any], table_name: str = "krishimitra.disease_predictions_log"):
    """Log a disease prediction result to Delta Lake."""
    spark = get_spark()
    from pyspark.sql import Row

    row = Row(
        prediction_id=str(uuid.uuid4()),
        timestamp=datetime.now().isoformat(),
        predicted_disease=prediction.get("disease_raw", ""),
        predicted_crop=prediction.get("crop", ""),
        confidence=float(prediction.get("confidence", 0.0)),
        treatment=prediction.get("treatment", ""),
        model_version=prediction.get("model_version", "latest"),
        user_language=prediction.get("language", "en"),
    )

    df = spark.createDataFrame([row])

    try:
        df.write.format("delta").mode("append").saveAsTable(table_name)
        logger.info(f"✅ Logged prediction {row.prediction_id}")
    except Exception as e:
        logger.warning(f"Could not log prediction to Delta: {e}")


def log_price_prediction(prediction: Dict[str, Any], table_name: str = "krishimitra.price_predictions_log"):
    """Log a price prediction result to Delta Lake."""
    spark = get_spark()
    from pyspark.sql import Row

    row = Row(
        prediction_id=str(uuid.uuid4()),
        timestamp=datetime.now().isoformat(),
        commodity=prediction.get("commodity", ""),
        market=prediction.get("market", ""),
        state=prediction.get("state", ""),
        predicted_price=float(prediction.get("predicted_price", 0.0)),
        days_ahead=int(prediction.get("days_ahead", 7)),
        model_version=prediction.get("model_version", "latest"),
    )

    df = spark.createDataFrame([row])

    try:
        df.write.format("delta").mode("append").saveAsTable(table_name)
        logger.info(f"✅ Logged price prediction {row.prediction_id}")
    except Exception as e:
        logger.warning(f"Could not log price prediction to Delta: {e}")


def log_chat_session(session_data: Dict[str, Any], table_name: str = "krishimitra.chat_sessions"):
    """Log a chat session interaction to Delta Lake."""
    spark = get_spark()
    from pyspark.sql import Row

    row = Row(
        session_id=str(uuid.uuid4()),
        timestamp=datetime.now().isoformat(),
        user_query=session_data.get("query", ""),
        intent=session_data.get("intent", ""),
        response_summary=session_data.get("response", "")[:500],
        language=session_data.get("language", "en"),
        feature_used=session_data.get("feature", ""),
    )

    df = spark.createDataFrame([row])

    try:
        df.write.format("delta").mode("append").saveAsTable(table_name)
    except Exception as e:
        logger.warning(f"Could not log chat session to Delta: {e}")


# ──────────────────────────────────────────────
# Table Info / Diagnostics
# ──────────────────────────────────────────────

def get_table_stats(table_name: str) -> Dict[str, Any]:
    """Get basic statistics for a Delta table."""
    try:
        spark = get_spark()
        df = spark.table(table_name)
        count = df.count()
        columns = df.columns
        return {
            "table": table_name,
            "row_count": count,
            "columns": columns,
            "column_count": len(columns),
        }
    except Exception as e:
        return {"table": table_name, "error": str(e)}


def list_tables(database: str = "krishimitra") -> List[str]:
    """List all tables in the database."""
    try:
        spark = get_spark()
        tables_df = spark.sql(f"SHOW TABLES IN {database}")
        return [row["tableName"] for row in tables_df.collect()]
    except Exception as e:
        logger.error(f"Could not list tables: {e}")
        return []
