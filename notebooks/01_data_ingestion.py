# Databricks notebook source
# MAGIC %md
# MAGIC ## KrishiMitra — Data Ingestion Pipeline (Bronze Layer)
# MAGIC
# MAGIC This notebook ingests raw agricultural data into Unity Catalog Delta tables
# MAGIC using **Auto Loader** for incremental processing.
# MAGIC
# MAGIC **Architecture:**
# MAGIC - **Source**: JSON files in `/Volumes/{catalog}/{schema}/data/`
# MAGIC - **Mandi prices table**: ~1.1M commodity price records from AgMarkNet (`price_id` as PK)
# MAGIC - **Government schemes table**: 24 central/state agricultural schemes
# MAGIC - **Crop knowledge table**: 38 disease entries for crop disease detection
# MAGIC - **Pesticide/fertilizer table**: 38 product entries across major crops
# MAGIC - Auto Loader handles schema inference, incremental file discovery, and exactly-once processing
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
source_path = f"/Volumes/{catalog}/{schema}/data"
checkpoint_base = f"{source_path}/_checkpoints"

spark.sql(f"USE CATALOG {catalog}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {schema}")
spark.sql(f"USE SCHEMA {schema}")

print(f"Target: {catalog}.{schema}")
print(f"Source: {source_path}")
print(f"Checkpoints: {checkpoint_base}")

# COMMAND ----------
# DBTITLE 1,Ingest Mandi Prices with Auto Loader

from pyspark.sql import functions as F
from pyspark.sql.types import *

# --- Mandi Prices Ingestion (JSON — AgMarkNet format) ---
# File: agmarknet_india_historical_prices_2024_2025.json
# Fields: "Sl no.", "District Name", "Market Name", "Commodity", "Variety",
#         "Grade", "Min Price (Rs./Quintal)", "Max Price (Rs./Quintal)",
#         "Modal Price (Rs./Quintal)", "Price Date", "State"
prices_stream = (
    spark.readStream.format("cloudFiles")
    .option("cloudFiles.format", "json")
    .option("multiLine", "true")
    .option("cloudFiles.inferColumnTypes", "true")
    .option("cloudFiles.schemaLocation", f"{checkpoint_base}/mandi_prices/schema")
    .option("pathGlobFilter", "*prices*.json")
    .load(source_path)
)

# Flatten and rename columns from AgMarkNet format to clean schema
prices_transformed = prices_stream.select(
    F.expr("uuid()").alias("price_id"),
    F.col("State").alias("state"),
    F.col("`District Name`").alias("district"),
    F.col("`Market Name`").alias("market"),
    F.col("Commodity").alias("commodity"),
    F.col("Variety").alias("variety"),
    F.col("Grade").alias("grade"),
    F.to_date(F.col("`Price Date`"), "dd MMM yyyy").alias("arrival_date"),
    F.col("`Min Price (Rs./Quintal)`").cast("double").alias("min_price"),
    F.col("`Max Price (Rs./Quintal)`").cast("double").alias("max_price"),
    F.col("`Modal Price (Rs./Quintal)`").cast("double").alias("modal_price"),
    F.col("_metadata.file_path").alias("source_file"),
)

# Write to Delta
query_prices = (
    prices_transformed.writeStream
    .option("checkpointLocation", f"{checkpoint_base}/mandi_prices/checkpoint")
    .option("mergeSchema", "true")
    .trigger(availableNow=True)
    .toTable(f"{catalog}.{schema}.mandi_prices_raw")
)

query_prices.awaitTermination()
print(f"Mandi prices ingested into {catalog}.{schema}.mandi_prices_raw")

# COMMAND ----------
# DBTITLE 1,Ingest Government Schemes with Auto Loader

from pyspark.sql import functions as F

# --- Government Schemes Ingestion (JSON) ---
# File: govt_schemes.json
# Fields: scheme_id, name, name_en, category, description, eligibility,
#         exclusion_criteria, benefits, how_to_apply, documents_required,
#         official_url, helpline, coverage
schemes_stream = (
    spark.readStream.format("cloudFiles")
    .option("cloudFiles.format", "json")
    .option("multiLine", "true")
    .option("cloudFiles.inferColumnTypes", "true")
    .option("cloudFiles.schemaLocation", f"{checkpoint_base}/govt_schemes/schema")
    .option("pathGlobFilter", "*schemes*.json")
    .load(source_path)
)

# Flatten and transform
schemes_transformed = schemes_stream.select(
    F.col("scheme_id"),
    F.col("name"),
    F.col("name_en"),
    F.col("category"),
    F.col("description"),
    F.col("eligibility"),
    F.col("exclusion_criteria"),
    F.col("benefits"),
    F.col("how_to_apply"),
    F.col("documents_required"),
    F.col("official_url"),
    F.col("helpline"),
    F.col("coverage"),
    F.col("_metadata.file_path").alias("source_file"),
)

# Write to Delta
query_schemes = (
    schemes_transformed.writeStream
    .option("checkpointLocation", f"{checkpoint_base}/govt_schemes/checkpoint")
    .option("mergeSchema", "true")
    .trigger(availableNow=True)
    .toTable(f"{catalog}.{schema}.govt_schemes")
)

query_schemes.awaitTermination()
print(f"Government schemes ingested into {catalog}.{schema}.govt_schemes")

# COMMAND ----------
# DBTITLE 1,Ingest Crop Knowledge with Auto Loader

from pyspark.sql import functions as F

# --- Crop Knowledge Ingestion (JSON) ---
# File: crop_knowledge.json
# Fields: disease_class, treatment, prevention, organic
# disease_class format: "Crop___Disease_name" e.g. "Tomato___Late_blight"
crop_stream = (
    spark.readStream.format("cloudFiles")
    .option("cloudFiles.format", "json")
    .option("multiLine", "true")
    .option("cloudFiles.inferColumnTypes", "true")
    .option("cloudFiles.schemaLocation", f"{checkpoint_base}/crop_knowledge/schema")
    .option("pathGlobFilter", "*crop_knowledge*.json")
    .load(source_path)
)

# Flatten and transform — derive crop and disease_name from disease_class
crop_transformed = crop_stream.select(
    F.col("disease_class"),
    # Extract crop name from "Crop___Disease" pattern
    F.regexp_extract("disease_class", r"^([^_]+)", 1).alias("crop"),
    # Extract disease name from "Crop___Disease_name" pattern, replace underscores
    F.regexp_replace(
        F.regexp_extract("disease_class", r"___(.+)$", 1),
        "_", " "
    ).alias("disease_name"),
    F.col("treatment"),
    F.col("prevention"),
    F.col("organic"),
    F.col("_metadata.file_path").alias("source_file"),
)

# Write to Delta
query_crop = (
    crop_transformed.writeStream
    .option("checkpointLocation", f"{checkpoint_base}/crop_knowledge/checkpoint")
    .option("mergeSchema", "true")
    .trigger(availableNow=True)
    .toTable(f"{catalog}.{schema}.crop_knowledge")
)

query_crop.awaitTermination()
print(f"Crop knowledge ingested into {catalog}.{schema}.crop_knowledge")

# COMMAND ----------
# DBTITLE 1,Ingest Pesticide/Fertilizer Guide with Auto Loader

from pyspark.sql import functions as F

# --- Pesticide/Fertilizer Guide Ingestion (JSON) ---
# File: pesticide_fertilizer.json
# Fields: crop, growth_stage, category, problem, product_name, dosage,
#         application_method, timing, precautions, organic_alternative, cost_estimate
pesticide_stream = (
    spark.readStream.format("cloudFiles")
    .option("cloudFiles.format", "json")
    .option("multiLine", "true")
    .option("cloudFiles.inferColumnTypes", "true")
    .option("cloudFiles.schemaLocation", f"{checkpoint_base}/pesticide_guide/schema")
    .option("pathGlobFilter", "*pesticide*.json")
    .load(source_path)
)

# Flatten and transform
pesticide_transformed = pesticide_stream.select(
    F.col("crop"),
    F.col("growth_stage"),
    F.col("category"),
    F.col("problem"),
    F.col("product_name"),
    F.col("dosage"),
    F.col("application_method"),
    F.col("timing"),
    F.col("precautions"),
    F.col("organic_alternative"),
    F.col("cost_estimate"),
    F.col("_metadata.file_path").alias("source_file"),
)

# Write to Delta
query_pest = (
    pesticide_transformed.writeStream
    .option("checkpointLocation", f"{checkpoint_base}/pesticide_guide/checkpoint")
    .option("mergeSchema", "true")
    .trigger(availableNow=True)
    .toTable(f"{catalog}.{schema}.pesticide_fertilizer_guide")
)

query_pest.awaitTermination()
print(f"Pesticide/fertilizer guide ingested into {catalog}.{schema}.pesticide_fertilizer_guide")

# COMMAND ----------
# DBTITLE 1,Establish PK Constraints

# PK columns must be NOT NULL — set nullability first, then add constraints
spark.sql(f"ALTER TABLE {catalog}.{schema}.mandi_prices_raw ALTER COLUMN price_id SET NOT NULL")

spark.sql(f"""
  ALTER TABLE {catalog}.{schema}.mandi_prices_raw
  ADD CONSTRAINT mandi_prices_pk PRIMARY KEY (price_id)
""")

# scheme_id is the natural key for government schemes
spark.sql(f"ALTER TABLE {catalog}.{schema}.govt_schemes ALTER COLUMN scheme_id SET NOT NULL")

spark.sql(f"""
  ALTER TABLE {catalog}.{schema}.govt_schemes
  ADD CONSTRAINT govt_schemes_pk PRIMARY KEY (scheme_id)
""")

print("PK constraints added successfully")

# COMMAND ----------
# DBTITLE 1,Enable Change Data Feed

# Enable CDF on all tables for incremental downstream reads via table_changes()
for tbl in ["mandi_prices_raw", "govt_schemes", "crop_knowledge", "pesticide_fertilizer_guide"]:
    spark.sql(f"ALTER TABLE {catalog}.{schema}.{tbl} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")
    print(f"Change Data Feed enabled on {catalog}.{schema}.{tbl}")

# COMMAND ----------
# DBTITLE 1,Add Table & Column Comments

# ── Table-level comments ──
spark.sql(f"""
  COMMENT ON TABLE {catalog}.{schema}.mandi_prices_raw IS
  'Raw commodity mandi prices from AgMarkNet (data.gov.in). Contains ~1.1M rows of daily modal/min/max prices across Indian agricultural markets for 2024-2025. Primary key: price_id. Ingested incrementally via Auto Loader from JSON files.'
""")

spark.sql(f"""
  COMMENT ON TABLE {catalog}.{schema}.govt_schemes IS
  'Indian government agricultural schemes for farmer welfare. Contains 24 central and state schemes with eligibility, benefits, exclusion criteria, and application details. Primary key: scheme_id. Used for RAG-based advisory.'
""")

spark.sql(f"""
  COMMENT ON TABLE {catalog}.{schema}.crop_knowledge IS
  'Crop disease knowledge base with 38 entries covering 14 crops. Maps disease_class (PlantVillage format: Crop___Disease) to treatment, prevention, and organic remedies. Used for post-disease-detection advisory.'
""")

spark.sql(f"""
  COMMENT ON TABLE {catalog}.{schema}.pesticide_fertilizer_guide IS
  'Pesticide and fertilizer guide with 38 entries across 8 crops. Covers growth-stage-specific recommendations including dosage, application method, timing, precautions, organic alternatives, and estimated costs.'
""")

# ── Mandi Prices column comments ──
prices_column_comments = {
    "price_id":       "Unique auto-generated UUID for each price record. Primary key.",
    "state":          "Indian state where the mandi/market is located (e.g. Uttar Pradesh, Maharashtra). From AgMarkNet 'State' field.",
    "district":       "District within the state where the market operates. From AgMarkNet 'District Name' field.",
    "market":         "Name of the agricultural mandi/market (e.g. Achalda, Azadpur). From AgMarkNet 'Market Name' field.",
    "commodity":      "Name of the traded commodity (e.g. Wheat, Tomato, Onion). From AgMarkNet 'Commodity' field.",
    "variety":        "Variety of the commodity (e.g. Dara, Local, Hybrid). From AgMarkNet 'Variety' field.",
    "grade":          "Quality grade of the commodity (e.g. FAQ - Fair Average Quality). From AgMarkNet 'Grade' field.",
    "arrival_date":   "Date when the commodity price was recorded. Parsed from AgMarkNet 'Price Date' field (dd MMM yyyy format, e.g. 05 Apr 2025).",
    "min_price":      "Minimum wholesale price per quintal (100 kg) in Indian Rupees (INR). From AgMarkNet 'Min Price (Rs./Quintal)' field.",
    "max_price":      "Maximum wholesale price per quintal in INR. From AgMarkNet 'Max Price (Rs./Quintal)' field.",
    "modal_price":    "Modal (most common) wholesale price per quintal in INR. Primary price indicator for predictions. From AgMarkNet 'Modal Price (Rs./Quintal)' field.",
    "source_file":    "Auto Loader metadata: full path of the source JSON file this record was ingested from.",
}

for col, comment in prices_column_comments.items():
    safe = comment.replace("'", "\\'")
    spark.sql(f"ALTER TABLE {catalog}.{schema}.mandi_prices_raw ALTER COLUMN {col} COMMENT '{safe}'")

# ── Government Schemes column comments ──
schemes_column_comments = {
    "scheme_id":            "Unique scheme identifier (e.g. SCHEME_001). Primary key.",
    "name":                 "Short scheme name/abbreviation (e.g. PM-KISAN, PMFBY).",
    "name_en":              "Full official scheme name in English.",
    "category":             "Scheme category. Values: crop_insurance, income_support, irrigation, credit, soil_health, organic_farming, horticulture, mechanization, marketing, livestock, fisheries.",
    "description":          "Detailed description of the scheme objectives and provisions.",
    "eligibility":          "Array of eligibility criteria strings that a farmer must satisfy.",
    "exclusion_criteria":   "Criteria for exclusion from the scheme (e.g. income tax payers, government employees).",
    "benefits":             "Description of financial and non-financial benefits provided by the scheme.",
    "how_to_apply":         "Step-by-step application process description.",
    "documents_required":   "Array of documents needed for application (e.g. Aadhaar, land records).",
    "official_url":         "Official government website URL for the scheme.",
    "helpline":             "Toll-free helpline number or contact info for scheme inquiries.",
    "coverage":             "Geographic coverage. Values: Pan India, All India, or specific state names.",
    "source_file":          "Auto Loader metadata: full path of the source JSON file this record was ingested from.",
}

for col, comment in schemes_column_comments.items():
    safe = comment.replace("'", "\\'")
    spark.sql(f"ALTER TABLE {catalog}.{schema}.govt_schemes ALTER COLUMN {col} COMMENT '{safe}'")

# ── Crop Knowledge column comments ──
crop_column_comments = {
    "disease_class":  "PlantVillage disease class identifier in format Crop___Disease_name (e.g. Tomato___Late_blight, Apple___Apple_scab, Corn_(maize)___Common_rust_). Used as key to match HuggingFace model predictions.",
    "crop":           "Crop name extracted from disease_class (e.g. Tomato, Apple, Corn). Derived field.",
    "disease_name":   "Human-readable disease name extracted from disease_class with underscores replaced by spaces (e.g. Late blight, Apple scab). Shows 'healthy' for healthy crop classes.",
    "treatment":      "Recommended chemical treatment (e.g. Apply Captan or Myclobutanil). Shows 'No treatment required' for healthy classes.",
    "prevention":     "Preventive agricultural practices (e.g. Rake up fallen leaves, prune trees for aeration).",
    "organic":        "Organic/natural treatment alternative (e.g. Liquid copper or sulfur, Bordeaux mixture).",
    "source_file":    "Auto Loader metadata: full path of the source JSON file this record was ingested from.",
}

for col, comment in crop_column_comments.items():
    safe = comment.replace("'", "\\'")
    spark.sql(f"ALTER TABLE {catalog}.{schema}.crop_knowledge ALTER COLUMN {col} COMMENT '{safe}'")

# ── Pesticide/Fertilizer Guide column comments ──
pesticide_column_comments = {
    "crop":                 "Target crop for this recommendation (e.g. Wheat, Rice, Tomato, Cotton, Soybean).",
    "growth_stage":         "Crop growth stage for application timing (e.g. Sowing, Tillering, Heading, Grain filling, Nursery, Transplanting).",
    "category":             "Product category. Values: fertilizer, herbicide, fungicide, pesticide, insecticide.",
    "problem":              "Specific pest, disease, or nutrient issue this product addresses (e.g. Nutrient deficiency, Yellow Rust, Aphids, Broadleaf weeds).",
    "product_name":         "Commercial or generic name of the recommended product (e.g. Urea & DAP, Propiconazole 25% EC, Imidacloprid 17.8% SL).",
    "dosage":               "Recommended dosage with units (e.g. 50 kg Urea per acre, 200 ml per acre, 250-300 grams per acre).",
    "application_method":   "How to apply the product (e.g. Basal dose application, Foliar spray, Seed treatment).",
    "timing":               "Best time for application (e.g. At the time of sowing, 30-35 days after sowing, Appearance of initial symptoms).",
    "precautions":          "Safety precautions (e.g. Avoid spraying during peak bee activity, Use flat fan nozzle).",
    "organic_alternative":  "Organic/natural alternative (e.g. FYM 5 tonnes/acre, Neem Oil 1500ppm, Manual weeding).",
    "cost_estimate":        "Estimated cost per acre in INR (e.g. Rs 1,200-1,300 per acre).",
    "source_file":          "Auto Loader metadata: full path of the source JSON file this record was ingested from.",
}

for col, comment in pesticide_column_comments.items():
    safe = comment.replace("'", "\\'")
    spark.sql(f"ALTER TABLE {catalog}.{schema}.pesticide_fertilizer_guide ALTER COLUMN {col} COMMENT '{safe}'")

print(f"Table and column comments added for all 4 tables")

# COMMAND ----------
# DBTITLE 1,Verify Ingested Data

from pyspark.sql import functions as F

# Row counts
prices_count = spark.table(f"{catalog}.{schema}.mandi_prices_raw").count()
schemes_count = spark.table(f"{catalog}.{schema}.govt_schemes").count()
crop_count = spark.table(f"{catalog}.{schema}.crop_knowledge").count()
pest_count = spark.table(f"{catalog}.{schema}.pesticide_fertilizer_guide").count()

print(f"Mandi Prices:       {prices_count:>10,} rows")
print(f"Government Schemes: {schemes_count:>10,} rows")
print(f"Crop Knowledge:     {crop_count:>10,} rows")
print(f"Pesticide Guide:    {pest_count:>10,} rows")

# Commodity distribution
print("\n--- Top Commodities by Records ---")
display(
    spark.table(f"{catalog}.{schema}.mandi_prices_raw")
    .groupBy("commodity")
    .agg(F.count("*").alias("records"), F.round(F.avg("modal_price"), 2).alias("avg_price"))
    .orderBy(F.desc("records"))
    .limit(10)
)

# Sample schemes
print("\n--- Sample Government Schemes ---")
display(
    spark.table(f"{catalog}.{schema}.govt_schemes")
    .select("scheme_id", "name_en", "category", "coverage")
    .limit(5)
)

# Sample crop knowledge
print("\n--- Sample Crop Knowledge ---")
display(
    spark.table(f"{catalog}.{schema}.crop_knowledge")
    .select("disease_class", "crop", "disease_name", "treatment")
    .limit(5)
)
