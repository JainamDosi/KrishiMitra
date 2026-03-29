"""
KrishiMitra Source Package
==========================
AI-Powered Farmer Assistance Platform

Modules:
    - disease_predictor: Crop disease detection via pre-trained HuggingFace model
    - price_predictor: Mandi price analytics & forecasting via Spark MLlib
    - scheme_advisor: Government scheme RAG advisory via FAISS + Sarvam AI
    - pesticide_advisor: Pesticide & fertilizer RAG advisory
    - chat_engine: Intent classification & conversation routing
    - translator: Sarvam AI translation wrapper (10+ Indian languages)
    - delta_utils: Delta Lake read/write utilities
"""

__version__ = "1.0.0"
__app_name__ = "KrishiMitra"

__all__ = [
    "disease_predictor",
    "price_predictor",
    "scheme_advisor",
    "pesticide_advisor",
    "chat_engine",
    "translator",
    "delta_utils",
]
