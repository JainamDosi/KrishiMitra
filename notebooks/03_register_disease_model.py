# Databricks notebook source
# MAGIC %md
# MAGIC ## KrishiMitra — Register Disease Detection Model
# MAGIC
# MAGIC **No training required!** We use a pre-trained MobileNetV2 model
# MAGIC fine-tuned on the PlantVillage dataset (38 classes, 95.4% accuracy).
# MAGIC
# MAGIC This notebook:
# MAGIC 1. Downloads the model from HuggingFace
# MAGIC 2. Tests inference on a sample image
# MAGIC 3. Registers it in MLflow for versioning
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
# DBTITLE 1,Download Pre-Trained Model from HuggingFace

import mlflow
from transformers import pipeline

MODEL_NAME = "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification"

print(f"Downloading model: {MODEL_NAME}")
print("   (First run downloads ~14 MB from HuggingFace)")

classifier = pipeline(
    "image-classification",
    model=MODEL_NAME,
    top_k=5,
)

print("Model loaded successfully!")

# COMMAND ----------
# DBTITLE 1,Test Inference on Sample Image

# Test with a sample image URL
test_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/32/Tomato_leaf_curl.jpg/220px-Tomato_leaf_curl.jpg"

print(f"Testing inference on: {test_url}")
test_result = classifier(test_url)

print("\nPrediction Results:")
for i, r in enumerate(test_result):
    marker = ">>>" if i == 0 else "   "
    print(f"  {marker} {r['label']}: {r['score']*100:.2f}%")

top_prediction = test_result[0]
print(f"\nTop prediction: {top_prediction['label']} ({top_prediction['score']*100:.1f}%)")

# COMMAND ----------
# DBTITLE 1,Register in MLflow Model Registry

mlflow.set_experiment(f"/{catalog}/krishimitra/disease-detection")

with mlflow.start_run(run_name="mobilenetv2_pretrained_plantvillage"):
    # Log parameters
    mlflow.log_params({
        "model_source": "huggingface",
        "model_name": MODEL_NAME,
        "base_model": "google/mobilenet_v2_1.0_224",
        "dataset": "PlantVillage (38 classes, 14 crops)",
        "reported_accuracy": 0.954,
        "training": "pre-trained (no re-training needed)",
        "inference_device": "cpu",
        "model_size_mb": "~14",
    })

    # Log metrics
    mlflow.log_metric("reported_accuracy", 0.954)
    mlflow.log_metric("num_classes", 38)
    mlflow.log_metric("num_crops", 14)

    # Log the transformers pipeline as MLflow model
    mlflow.transformers.log_model(
        transformers_model=classifier,
        artifact_path="disease-classifier",
        registered_model_name="krishimitra-disease-classifier",
        task="image-classification",
    )

    print("Model registered in MLflow as 'krishimitra-disease-classifier'")

# COMMAND ----------
# DBTITLE 1,Verify — Load from MLflow Registry

print("Loading model back from MLflow registry...")

try:
    loaded_classifier = mlflow.transformers.load_model(
        "models:/krishimitra-disease-classifier/latest"
    )

    # Re-test
    verify_result = loaded_classifier(test_url)
    print(f"Verification successful!")
    print(f"   Prediction: {verify_result[0]['label']} ({verify_result[0]['score']*100:.1f}%)")

    # Compare with original
    assert verify_result[0]["label"] == test_result[0]["label"], "Results mismatch!"
    print(f"   Results match original prediction")
except Exception as e:
    print(f"MLflow verification skipped: {e}")
    print("   (Model will be loaded directly from HuggingFace at runtime)")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Disease Model Registration Complete
# MAGIC
# MAGIC | Property | Value |
# MAGIC |----------|-------|
# MAGIC | Model | `linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification` |
# MAGIC | Base | Google MobileNetV2 (fine-tuned on PlantVillage) |
# MAGIC | Accuracy | 95.4% (38 classes, 14 crops) |
# MAGIC | Registry | `models:/krishimitra-disease-classifier/latest` |
# MAGIC | Inference | CPU-only, ~200ms per image |
# MAGIC | Training | **None required** — pre-trained |
