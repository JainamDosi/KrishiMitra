# 🌾 KrishiMitra — AI-Powered Agricultural Advisory Platform

End-to-end data + AI pipeline providing Indian farmers with crop disease detection, market price predictions, government scheme discovery, and pesticide/fertilizer recommendations — all in 10+ Indian languages.

## Architecture

```
data/raw/                       JSON files (AgMarkNet prices, schemes, crop KB, pesticides)
│
▼
01-data-ingestion               Auto Loader → bronze Delta tables (4 tables)
│
▼
02-data-transformation          PySpark ETL + DLT SQL → silver → gold tables
│
├──────────────────┬─────────────────┬──────────────────┐
▼                  ▼                 ▼                  ▼
03-disease-model   04-price-model    05-vector-index    02-DLT-pipeline
(HuggingFace →     (Spark MLlib →    (FAISS →           (SQL transforms →
 MLflow)            MLflow)           UC Volume)         silver/gold)
│                  │                 │
└──────────────────┴─────────────────┘
                   │
                   ▼
          06-demo-walkthrough       Full feature verification
                   │
                   ▼
          Databricks App            FastAPI + HTML/CSS/JS UI
          (server.py + public/)     with Sarvam AI multilingual
```

## Contents

```
krishimitra/
├── data/raw/                                    # Raw data files for ingestion
│   ├── agmarknet_india_historical_prices_2024_2025.json   # ~1.1M mandi price records (93 MB)
│   ├── govt_schemes.json                                  # 24 government agricultural schemes
│   ├── crop_knowledge.json                                # 38 crop disease entries (PlantVillage)
│   └── pesticide_fertilizer.json                          # 38 pesticide/fertilizer entries
│
├── notebooks/
│   ├── 01_data_ingestion.py                     # Auto Loader → 4 bronze Delta tables
│   ├── 02_delta_lake_etl.py                     # PySpark ETL → enriched tables + logging tables
│   ├── 02_data_transformation/
│   │   └── transformations/
│   │       ├── silver_mandi_prices.sql          # DLT: cleaned prices with quality constraints
│   │       ├── gold_price_analytics.sql         # DLT: aggregated price analytics view
│   │       └── gold_scheme_chunks.sql           # DLT: chunked schemes for RAG embedding
│   ├── 03_register_disease_model.py             # HuggingFace MobileNetV2 → MLflow
│   ├── 04_price_prediction_model.py             # Spark MLlib GBTRegressor → MLflow
│   ├── 05_vector_index_build.py                 # FAISS vector indexes → UC Volume
│   └── 06_demo_walkthrough.py                   # End-to-end demo for judges
│
├── src/                                         # Backend application modules
│   ├── chat_engine.py                           # Intent detection + response routing
│   ├── disease_predictor.py                     # MobileNetV2 inference wrapper
│   ├── price_predictor.py                       # GBT price prediction wrapper
│   ├── scheme_advisor.py                        # FAISS-based scheme RAG
│   ├── pesticide_advisor.py                     # FAISS-based pesticide RAG
│   ├── translator.py                            # Sarvam AI multilingual translation
│   └── delta_utils.py                           # Delta Lake read/write utilities
│
├── public/                                      # Frontend web application
│   ├── index.html                               # Main UI layout
│   ├── style.css                                # Styling
│   └── script.js                                # Client-side logic
│
├── server.py                                    # FastAPI application server
├── app.yaml                                     # Databricks App deployment config
├── requirements.txt                             # Python dependencies
└── .env.example                                 # Environment variable template
```

### Tables (Unity Catalog)

| Table | Primary Key | Description |
|-------|-------------|-------------|
| `mandi_prices_raw` | `price_id` | Raw commodity prices from AgMarkNet JSON (~1.1M rows) |
| `mandi_prices` | `price_id` | Enriched prices with moving averages, volatility, % change |
| `govt_schemes` | `scheme_id` | 24 government agricultural schemes |
| `crop_knowledge` | `disease_class` | 38 crop disease entries (PlantVillage format) |
| `pesticide_fertilizer_guide` | — | 38 pesticide/fertilizer product entries |
| `disease_predictions_log` | — | Log of disease detection predictions |
| `price_predictions_log` | — | Log of price prediction requests |
| `chat_sessions` | — | Log of user chat interactions |

### MLflow Models

| Model | Algorithm | Registry Name |
|-------|-----------|---------------|
| Disease Classifier | MobileNetV2 (pre-trained, PlantVillage) | `krishimitra-disease-classifier` |
| Price Predictor | GBTRegressor (Spark MLlib Pipeline) | `krishimitra-price-predictor` |

### FAISS Vector Indexes

| Index | Vectors | Dimension | Storage |
|-------|---------|-----------|---------|
| Scheme FAISS | 2 chunks/scheme | 384 | `/Volumes/{catalog}/{schema}/models/scheme_faiss.index` |
| Pesticide FAISS | 1 chunk/entry | 384 | `/Volumes/{catalog}/{schema}/models/pesticide_faiss.index` |

**Embedding Model**: `all-MiniLM-L6-v2` (384-dim)

---

## Prerequisites

- Databricks workspace with Unity Catalog (Free Edition works!)
- GitHub account with access to fork/clone this repository
- GitHub Personal Access Token (classic) with `repo` scope for Git integration
- SQL warehouse (Pro or Serverless) — Free Edition includes Starter warehouse
- **API Keys** (for the web application):
  - [Sarvam AI](https://www.sarvam.ai/) API key (multilingual translation)
  - [data.gov.in](https://data.gov.in/) API key (optional, for live price data)

---

## Setup Guide

### Step 1: Get a Databricks Workspace

If you don't have one, sign up for [Databricks Free Edition](https://www.databricks.com/try-databricks-free):
1. Search "databricks free edition" or visit the link above
2. Click **"Get started free"** and complete registration
3. Verify your email and log in

### Step 2: Create Catalog, Schema & Volumes

In **Databricks SQL Editor**, run:

```sql
CREATE CATALOG IF NOT EXISTS krishimitra;
CREATE SCHEMA IF NOT EXISTS krishimitra.agri_advisory;
```

Then create volumes via **Catalog UI**: Catalog → krishimitra → agri_advisory → Create → Volume:

```sql
-- Or run these SQL commands:
CREATE VOLUME IF NOT EXISTS krishimitra.agri_advisory.data;
CREATE VOLUME IF NOT EXISTS krishimitra.agri_advisory.models;
```

You should now have:
- Catalog: `krishimitra`
- Schema: `krishimitra.agri_advisory`
- Volume: `krishimitra.agri_advisory.data` (for raw data files)
- Volume: `krishimitra.agri_advisory.models` (for FAISS indexes)

### Step 3: Upload Raw Data

Upload the data files from `data/raw/` to your volume:

| File | Size | Upload To |
|------|------|-----------|
| `agmarknet_india_historical_prices_2024_2025.json` | 93 MB | `/Volumes/krishimitra/agri_advisory/data/` |
| `govt_schemes.json` | 44 KB | `/Volumes/krishimitra/agri_advisory/data/` |
| `crop_knowledge.json` | 10 KB | `/Volumes/krishimitra/agri_advisory/data/` |
| `pesticide_fertilizer.json` | 20 KB | `/Volumes/krishimitra/agri_advisory/data/` |

**How to upload:**
1. In Databricks: **Catalog** → `krishimitra` → `agri_advisory` → `data` → **Upload to this volume**
2. Drag and drop all 4 files

### Step 4: Connect Git Repository

#### 4.1: Generate GitHub Personal Access Token
1. Go to [GitHub.com](https://github.com) and log in
2. Navigate to **Settings** → **Developer settings** → **Personal access tokens** → **Tokens (classic)**
3. Click **Generate new token**
4. Set expiration and select `repo` scope (full repository access)
5. **Copy the generated token** (you won't see it again!)

#### 4.2: Add Git Credentials in Databricks
1. In your Databricks workspace, click on your **profile icon** (top right)
2. Select **Settings** → **Linked accounts** → **Add Git credential**
3. Choose **Personal access token** as authentication method
4. Paste your GitHub token and save

#### 4.3: Create Git Folder
1. Navigate to **Workspace** → **Create** → **Git folder**
2. Enter your repository URL: `https://github.com/<your-username>/krishimitra.git`
3. Click **Create Git folder**
4. Your repository files will now be accessible in the Databricks workspace

✅ **Success Check**: After creating the Git folder, you should see your repository under **Workspace → Repos**. Click on it to access all project files and notebooks directly within Databricks.

### Step 5: Create a Compute Cluster

1. Go to **Compute** → **Create compute**
2. Configure:
   - **Runtime**: `14.3 LTS ML` (or newer ML Runtime)
   - **Node type**: `Standard_DS3_v2` or equivalent (4 cores, 14 GB RAM)
   - **Access mode**: `Single User` (required for Unity Catalog)
   - **Auto-terminate**: 60 minutes
3. Click **Create compute**

> **Why ML Runtime?** It includes pre-installed `mlflow`, `transformers`, `torch`, and `plotly` — saving you from manual library installation.

### Step 6: Install Additional Libraries

From the cluster UI → **Libraries** → **Install New** → **PyPI**:

| Library | Purpose |
|---------|---------|
| `sentence-transformers` | Embedding model for FAISS vector search |
| `faiss-cpu` | Vector similarity search for RAG |

> `transformers`, `mlflow`, `torch`, and `plotly` come pre-installed on ML Runtime.

---

## Running the Notebooks

Run notebooks **in order** using the widget inputs at the top to set your catalog/schema:

### Step 7: Data Ingestion (Notebook 01)

1. Open `notebooks/01_data_ingestion.py`
2. Set widgets at the top:
   - **Catalog Name**: `krishimitra`
   - **Schema Name**: `agri_advisory`
3. **Run All** cells
4. **What it does**:
   - Auto Loader ingests JSON files: prices (~1.1M records) + schemes + crop knowledge + pesticides
   - Creates 4 bronze Delta tables with PK constraints
   - Enables Change Data Feed on all tables
   - Adds table + column comments for governance
5. **Expected time**: ~3–5 minutes
6. **Verify**: Last cell prints row counts for all 4 tables

### Step 8: Delta Lake ETL (Notebook 02)

1. Open `notebooks/02_delta_lake_etl.py`
2. Set same widget values: `krishimitra` / `agri_advisory`
3. **Run All** cells
4. **What it does**:
   - Reads `mandi_prices_raw` from bronze layer
   - Applies PySpark window functions (lag, 7/30-day moving averages, volatility)
   - Writes enriched `mandi_prices` table
   - Creates 3 logging tables (disease_predictions, price_predictions, chat_sessions)
5. **Expected time**: ~5–8 minutes
6. **Verify**: Last cell prints row counts for all 8 tables

### Step 9: (Optional) Create DLT Pipeline

If you want to use Delta Live Tables for the SQL transformations:

1. Go to **Workflows** → **Delta Live Tables** → **Create pipeline**
2. Configure:
   - **Source**: `notebooks/02_data_transformation/transformations/`
   - **Target catalog**: `krishimitra`
   - **Target schema**: `agri_advisory`
   - **Pipeline mode**: `Triggered`
3. Click **Start**
4. **What it creates**:
   - `silver_mandi_prices` — Streaming table with quality constraints
   - `gold_price_analytics` — Materialized view for aggregated analytics
   - `gold_scheme_chunks` — Materialized view chunking schemes for RAG

### Step 10: Disease Model Registration (Notebook 03)

1. Open `notebooks/03_register_disease_model.py`
2. Set widget values
3. **Run All** cells
4. **What it does**:
   - Downloads MobileNetV2 from HuggingFace (~14 MB, pre-trained on PlantVillage)
   - Tests inference on a sample tomato leaf curl image
   - Registers model in MLflow as `krishimitra-disease-classifier`
5. **Expected time**: ~2–3 minutes
6. **Verify**: MLflow experiment shows under `/{catalog}/krishimitra/disease-detection`

> ℹ️ **No training required!** This model is pre-trained with 95.4% accuracy across 38 disease classes.

### Step 11: Price Prediction Model (Notebook 04)

1. Open `notebooks/04_price_prediction_model.py`
2. Set widget values
3. **Run All** cells
4. **What it does**:
   - Loads enriched `mandi_prices` from Delta Lake
   - Builds Spark MLlib pipeline: StringIndexer → VectorAssembler → StandardScaler → GBTRegressor
   - Trains on 80/20 split, logs RMSE/MAE/R² to MLflow
   - Registers model as `krishimitra-price-predictor`
5. **Expected time**: ~5–10 minutes
6. **Verify**: MLflow experiment shows under `/{catalog}/krishimitra/price-prediction`

> ℹ️ **Steps 10, 11, and 12 can run in parallel** — they are independent of each other. They all depend on Step 8 (ETL) being complete.

### Step 12: Vector Index Build (Notebook 05)

1. Open `notebooks/05_vector_index_build.py`
2. Set widget values
3. **Run All** cells
4. **What it does**:
   - Loads schemes and pesticides from Delta tables (not raw JSON)
   - Generates embeddings using `all-MiniLM-L6-v2` (384-dim)
   - Builds FAISS indexes for both datasets
   - Saves indexes to UC Volume: `/Volumes/krishimitra/agri_advisory/models/`
5. **Expected time**: ~2–3 minutes
6. **Verify**: Test search results for "PM-KISAN" and "rice nitrogen" shown in output

### Step 13: Demo Walkthrough (Notebook 06)

1. Open `notebooks/06_demo_walkthrough.py`
2. Set widget values
3. **Run All** cells
4. **What it does**:
   - Verifies all 8 Delta tables exist with data
   - Demonstrates Delta Lake time-travel
   - Runs disease detection on sample image
   - Shows price trend chart with moving averages (Plotly)
   - Tests scheme RAG search
   - Tests pesticide RAG search
5. **Expected time**: ~3–5 minutes

> 🎯 **This is your demo notebook for judges!** Run this to showcase the full Databricks stack.

---

## Deploying the Web Application (Optional)

### Step 14: Configure Secrets

```bash
# Using Databricks CLI
databricks secrets create-scope krishimitra
databricks secrets put-secret krishimitra SARVAM_API_KEY --string-value "your_key_here"
databricks secrets put-secret krishimitra DATA_GOV_API_KEY --string-value "your_key_here"
```

### Step 15: Deploy as Databricks App

1. Upload `server.py`, `src/`, `public/`, `requirements.txt`, and `app.yaml` to your Workspace
2. Go to **Compute** → **Apps** → **Create App**
3. Configure:
   - **Source path**: Path to your app folder in Workspace
   - **App config**: `app.yaml` (auto-detected)
4. Click **Deploy**

The app provides a chat-based interface where farmers can:
- 📸 Upload crop leaf images for disease detection
- 📊 Get commodity price predictions and trends
- 🏛️ Ask about government schemes in their language
- 🧪 Get pesticide/fertilizer recommendations
- 🌐 Interact in 10+ Indian languages (via Sarvam AI)

---

## Execution DAG

```
┌─────────────────────┐
│  Step 2: Create     │
│  Catalog + Schema   │
│  + Volumes          │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Step 3: Upload     │
│  Raw Data (4 files) │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Step 7: Notebook   │
│  01_data_ingestion  │
│  (Auto Loader →     │
│   bronze tables)    │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Step 8: Notebook   │
│  02_delta_lake_etl  │
│  (PySpark ETL)      │
└─────────┬───────────┘
          │
    ┌─────┼──────────────┐
    │     │              │
    ▼     ▼              ▼
┌───────┐ ┌───────┐ ┌──────────┐
│ NB 03 │ │ NB 04 │ │  NB 05   │
│Disease│ │ Price │ │  Vector  │
│ Model │ │ Model │ │  Index   │
└───┬───┘ └───┬───┘ └────┬─────┘
    │         │           │
    └─────────┼───────────┘
              │
              ▼
    ┌─────────────────────┐
    │  Step 13: Notebook  │
    │  06_demo_walkthrough│
    │  (verification)     │
    └─────────┬───────────┘
              │
              ▼
    ┌─────────────────────┐
    │  Step 15: Deploy    │
    │  Databricks App     │
    │  (Optional)         │
    └─────────────────────┘
```

## Databricks Stack Usage

| Component | How Used |
|-----------|----------|
| **Unity Catalog** | Catalog + schema governance, Volumes for data + models |
| **Auto Loader** | Incremental ingestion of JSON files with schema inference |
| **Delta Lake** | 8 tables with PK constraints, CDF, time-travel, column comments |
| **PySpark** | Window functions (lag, moving avg), aggregations, feature engineering |
| **Delta Live Tables** | SQL streaming tables + materialized views with quality constraints |
| **Spark MLlib** | GBTRegressor pipeline with StringIndexer, VectorAssembler, StandardScaler |
| **MLflow** | Model registry, experiment tracking, metric logging (2 models) |
| **FAISS** | Vector search for RAG (government schemes + pesticide/fertilizer) |
| **Databricks App** | FastAPI + HTML/CSS/JS web application deployment |
| **Sarvam AI** | Multilingual translation (10+ Indian languages) |

## Features

| Feature | Description | Databricks Tech |
|---------|-------------|-----------------|
| 🦠 **Disease Detection** | Upload leaf image → identify disease (38 classes, 95.4% accuracy) → get treatment | HuggingFace MobileNetV2 + MLflow |
| 📊 **Price Prediction** | Select commodity/market → 7-day forecast with trend charts | Spark MLlib GBTRegressor + Delta Lake |
| 🏛️ **Scheme Advisory** | Ask about government schemes → RAG retrieval in any language | FAISS + Sarvam AI + Delta Lake |
| 🧪 **Pesticide Guide** | Describe crop problem → get product recommendations + organic alternatives | FAISS RAG + Delta Lake |
| 🌐 **Multilingual** | Interact in Hindi, Tamil, Telugu, Bengali, Marathi, and 5+ more languages | Sarvam AI Translation API |
| ⏰ **Time-Travel** | Query historical data versions for auditing and debugging | Delta Lake versioning |

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `CATALOG_NOT_FOUND` | Run Step 2 SQL commands to create catalog/schema |
| `VOLUME_NOT_FOUND` | Create volumes via Catalog UI or SQL (`CREATE VOLUME`) |
| Auto Loader finds no files | Ensure files are in `/Volumes/krishimitra/agri_advisory/data/` |
| `ModuleNotFoundError: faiss` | Install `faiss-cpu` via cluster Libraries tab |
| `ModuleNotFoundError: sentence_transformers` | Install `sentence-transformers` via cluster Libraries tab |
| MLflow model not found | Run notebook 03/04 first to register models |
| FAISS index not found | Run notebook 05 first to build indexes |
| Widget not showing | Click ⚙️ gear icon at top of notebook to see widget inputs |

---

## License

This project was built for the **Databricks Hackathon** (Digital Artha track).
