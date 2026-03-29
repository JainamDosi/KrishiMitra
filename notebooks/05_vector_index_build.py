# Databricks notebook source
# MAGIC %md
# MAGIC ## KrishiMitra — Vector Index Build (FAISS RAG Pipeline)
# MAGIC
# MAGIC Builds FAISS vector indexes for RAG-based advisory:
# MAGIC 1. Government schemes → `scheme_faiss.index`
# MAGIC 2. Pesticide/fertilizer guide → `pesticide_faiss.index`
# MAGIC
# MAGIC Uses `all-MiniLM-L6-v2` sentence transformer for embeddings (384-dim).
# MAGIC
# MAGIC **Configuration**: Use the widget inputs at the top of this notebook to set your catalog and schema.

# COMMAND ----------
# DBTITLE 1,Configuration & Setup

dbutils.widgets.text("catalog", "krishimitra", "Catalog Name")
dbutils.widgets.text("schema", "agri_advisory", "Schema Name")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
models_volume = f"/Volumes/{catalog}/{schema}/models"

spark.sql(f"USE CATALOG {catalog}")
spark.sql(f"USE SCHEMA {schema}")

print(f"Target: {catalog}.{schema}")
print(f"Models volume: {models_volume}")

# Ensure models volume directory exists
dbutils.fs.mkdirs(models_volume)

# COMMAND ----------
# DBTITLE 1,Load Sentence Transformer

import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

print("Loading sentence transformer: all-MiniLM-L6-v2")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
print(f"Model loaded. Embedding dimension: {embed_model.get_sentence_embedding_dimension()}")

# COMMAND ----------
# DBTITLE 1,Load Schemes from Delta Lake

# Read from Delta table (created by notebook 01) instead of raw JSON
schemes_df = spark.table(f"{catalog}.{schema}.govt_schemes")
schemes = [row.asDict() for row in schemes_df.collect()]
print(f"Loaded {len(schemes)} schemes from {catalog}.{schema}.govt_schemes")

# COMMAND ----------
# DBTITLE 1,Build Scheme FAISS Index

if schemes:
    # Create text chunks
    scheme_chunks = []
    for scheme in schemes:
        scheme_id = scheme.get("scheme_id", "")
        name = scheme.get("name_en", scheme.get("name", ""))

        # Chunk 1: Overview
        eligibility = scheme.get("eligibility", [])
        if isinstance(eligibility, list):
            eligibility = "; ".join(str(e) for e in eligibility)
        documents = scheme.get("documents_required", [])
        if isinstance(documents, list):
            documents = ", ".join(str(d) for d in documents)

        overview = (
            f"Scheme: {name}\n"
            f"Category: {scheme.get('category', '')}\n"
            f"Description: {scheme.get('description', '')}\n"
            f"Benefits: {scheme.get('benefits', '')}\n"
            f"Coverage: {scheme.get('coverage', '')}"
        )
        scheme_chunks.append({
            "scheme_id": scheme_id,
            "name": name,
            "chunk_type": "overview",
            "text": overview,
        })

        # Chunk 2: Application
        apply_text = (
            f"Scheme: {name}\n"
            f"Eligibility: {eligibility}\n"
            f"How to Apply: {scheme.get('how_to_apply', '')}\n"
            f"Documents Required: {documents}\n"
            f"Helpline: {scheme.get('helpline', '')}\n"
            f"Official URL: {scheme.get('official_url', '')}"
        )
        scheme_chunks.append({
            "scheme_id": scheme_id,
            "name": name,
            "chunk_type": "application",
            "text": apply_text,
        })

    print(f"Created {len(scheme_chunks)} scheme chunks")

    # Generate embeddings
    print("Generating scheme embeddings...")
    scheme_texts = [c["text"] for c in scheme_chunks]
    scheme_embeddings = embed_model.encode(scheme_texts, show_progress_bar=True)
    scheme_embeddings_np = np.array(scheme_embeddings).astype("float32")
    faiss.normalize_L2(scheme_embeddings_np)

    # Build FAISS index
    scheme_index = faiss.IndexFlatIP(scheme_embeddings_np.shape[1])
    scheme_index.add(scheme_embeddings_np)

    # Save to UC Volume
    scheme_index_path = f"/dbfs{models_volume}/scheme_faiss.index"
    scheme_chunks_path = f"/dbfs{models_volume}/scheme_chunks.json"

    os.makedirs(os.path.dirname(scheme_index_path), exist_ok=True)
    faiss.write_index(scheme_index, scheme_index_path)
    with open(scheme_chunks_path, "w", encoding="utf-8") as f:
        json.dump(scheme_chunks, f, ensure_ascii=False, indent=2)

    print(f"Scheme FAISS index saved: {models_volume}/scheme_faiss.index")
    print(f"   Vectors: {scheme_index.ntotal}")
    print(f"   Dimension: {scheme_embeddings_np.shape[1]}")

# COMMAND ----------
# DBTITLE 1,Load Pesticide Guide from Delta Lake

# Read from Delta table (created by notebook 01) instead of raw JSON
pest_df = spark.table(f"{catalog}.{schema}.pesticide_fertilizer_guide")
guide = [row.asDict() for row in pest_df.collect()]
print(f"Loaded {len(guide)} pesticide/fertilizer entries from {catalog}.{schema}.pesticide_fertilizer_guide")

# COMMAND ----------
# DBTITLE 1,Build Pesticide/Fertilizer FAISS Index

if guide:
    # Create text chunks
    pest_chunks = []
    for entry in guide:
        text = (
            f"Crop: {entry.get('crop', '')}\n"
            f"Growth Stage: {entry.get('growth_stage', '')}\n"
            f"Category: {entry.get('category', '')}\n"
            f"Problem: {entry.get('problem', '')}\n"
            f"Product: {entry.get('product_name', '')}\n"
            f"Dosage: {entry.get('dosage', '')}\n"
            f"Application Method: {entry.get('application_method', '')}\n"
            f"Timing: {entry.get('timing', '')}\n"
            f"Precautions: {entry.get('precautions', '')}\n"
            f"Organic Alternative: {entry.get('organic_alternative', '')}\n"
            f"Cost Estimate: {entry.get('cost_estimate', '')}"
        )
        pest_chunks.append({
            "crop": entry.get("crop", ""),
            "growth_stage": entry.get("growth_stage", ""),
            "category": entry.get("category", ""),
            "text": text,
        })

    print(f"Created {len(pest_chunks)} pesticide chunks")

    # Generate embeddings
    print("Generating pesticide embeddings...")
    pest_texts = [c["text"] for c in pest_chunks]
    pest_embeddings = embed_model.encode(pest_texts, show_progress_bar=True)
    pest_embeddings_np = np.array(pest_embeddings).astype("float32")
    faiss.normalize_L2(pest_embeddings_np)

    # Build FAISS index
    pest_index = faiss.IndexFlatIP(pest_embeddings_np.shape[1])
    pest_index.add(pest_embeddings_np)

    # Save to UC Volume
    pest_index_path = f"/dbfs{models_volume}/pesticide_faiss.index"
    pest_chunks_path = f"/dbfs{models_volume}/pesticide_chunks.json"

    faiss.write_index(pest_index, pest_index_path)
    with open(pest_chunks_path, "w", encoding="utf-8") as f:
        json.dump(pest_chunks, f, ensure_ascii=False, indent=2)

    print(f"Pesticide FAISS index saved: {models_volume}/pesticide_faiss.index")
    print(f"   Vectors: {pest_index.ntotal}")
    print(f"   Dimension: {pest_embeddings_np.shape[1]}")

# COMMAND ----------
# DBTITLE 1,Test Vector Search — Schemes

if schemes:
    print("Test: Scheme search for 'PM-KISAN farmer support'")
    test_query = "PM-KISAN farmer income support"
    q_emb = embed_model.encode(test_query).reshape(1, -1).astype("float32")
    faiss.normalize_L2(q_emb)
    scores, indices = scheme_index.search(q_emb, 3)

    for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
        print(f"\n  Result {i+1} (score: {score:.4f}):")
        print(f"  Scheme: {scheme_chunks[idx]['name']}")
        print(f"  Type: {scheme_chunks[idx]['chunk_type']}")
        print(f"  Text: {scheme_chunks[idx]['text'][:150]}...")

# COMMAND ----------
# DBTITLE 1,Test Vector Search — Pesticide

if guide:
    print("Test: Pesticide search for 'rice tillering nitrogen'")
    test_query = "rice tillering stage nitrogen deficiency"
    q_emb = embed_model.encode(test_query).reshape(1, -1).astype("float32")
    faiss.normalize_L2(q_emb)
    scores, indices = pest_index.search(q_emb, 3)

    for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
        print(f"\n  Result {i+1} (score: {score:.4f}):")
        print(f"  Crop: {pest_chunks[idx]['crop']}")
        print(f"  Stage: {pest_chunks[idx]['growth_stage']}")
        print(f"  Text: {pest_chunks[idx]['text'][:150]}...")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Vector Index Build Complete
# MAGIC
# MAGIC | Index | Vectors | Dimension | Path |
# MAGIC |-------|---------|-----------|------|
# MAGIC | Scheme FAISS | 2 chunks/scheme | 384 | `/Volumes/{catalog}/{schema}/models/scheme_faiss.index` |
# MAGIC | Pesticide FAISS | 1 chunk/entry | 384 | `/Volumes/{catalog}/{schema}/models/pesticide_faiss.index` |
# MAGIC
# MAGIC **Embedding Model**: `all-MiniLM-L6-v2` (384-dim, ~80 MB)
