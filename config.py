"""
KrishiMitra Configuration
========================
Central configuration for API keys, paths, model names, and constants.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ──────────────────────────────────────────────
# API Keys
# ──────────────────────────────────────────────
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "")
DATA_GOV_API_KEY = os.getenv("DATA_GOV_API_KEY", "")

# ──────────────────────────────────────────────
# Sarvam AI (Indian LLM 🇮🇳)
# ──────────────────────────────────────────────
SARVAM_BASE_URL = "https://api.sarvam.ai/v1"
SARVAM_CHAT_MODEL = "sarvam-m"
SARVAM_TRANSLATE_ENDPOINT = f"{SARVAM_BASE_URL}/translate"
SARVAM_CHAT_ENDPOINT = f"{SARVAM_BASE_URL}/chat/completions"

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# ──────────────────────────────────────────────
# Delta Lake Database
# ──────────────────────────────────────────────
DELTA_DB = "krishimitra_catalog.core_data"
DELTA_TABLES = {
    "mandi_prices": f"{DELTA_DB}.mandi_prices",
    "disease_log": f"{DELTA_DB}.disease_predictions_log",
    "price_log": f"{DELTA_DB}.price_predictions_log",
    "govt_schemes": f"{DELTA_DB}.govt_schemes",
    "crop_knowledge": f"{DELTA_DB}.crop_knowledge",
    "pesticide_guide": f"{DELTA_DB}.pesticide_fertilizer_guide",
    "chat_sessions": f"{DELTA_DB}.chat_sessions",
}

# ──────────────────────────────────────────────
# Disease Detection Model (Pre-trained HuggingFace)
# ──────────────────────────────────────────────
DISEASE_MODEL_HF = "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification"
DISEASE_MODEL_MLFLOW = "krishimitra-disease-classifier"
DISEASE_CLASSES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]

# ──────────────────────────────────────────────
# Price Prediction Model
# ──────────────────────────────────────────────
PRICE_MODEL_MLFLOW = "krishimitra-price-predictor"

# ──────────────────────────────────────────────
# Supported Languages (Sarvam AI)
# ──────────────────────────────────────────────
LANGUAGES = {
    "English": "en",
    "हिन्दी (Hindi)": "hi",
    "தமிழ் (Tamil)": "ta",
    "తెలుగు (Telugu)": "te",
    "ಕನ್ನಡ (Kannada)": "kn",
    "മലയാളം (Malayalam)": "ml",
    "मराठी (Marathi)": "mr",
    "বাংলা (Bengali)": "bn",
    "ગુજરાતી (Gujarati)": "gu",
    "ਪੰਜਾਬੀ (Punjabi)": "pa",
    "ଓଡ଼ିଆ (Odia)": "or",
}

# ──────────────────────────────────────────────
# FAISS Index Paths
# ──────────────────────────────────────────────
SCHEME_FAISS_INDEX = os.path.join(MODELS_DIR, "scheme_faiss.index")
SCHEME_CHUNKS_PATH = os.path.join(MODELS_DIR, "scheme_chunks.json")
PESTICIDE_FAISS_INDEX = os.path.join(MODELS_DIR, "pesticide_faiss.index")
PESTICIDE_CHUNKS_PATH = os.path.join(MODELS_DIR, "pesticide_chunks.json")

# ──────────────────────────────────────────────
# App Settings
# ──────────────────────────────────────────────
APP_NAME = "KrishiMitra 🌾"
APP_TAGLINE = "Har kisan ke liye, har sawaal ka jawaab"
APP_DESCRIPTION = "AI-Powered Farmer Assistance Platform"
GRADIO_SERVER_PORT = 7860
