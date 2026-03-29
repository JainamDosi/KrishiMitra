"""
KrishiMitra — Crop Disease Predictor
=====================================
Uses pre-trained HuggingFace MobileNetV2 model to detect 38 plant diseases
across 14 crop species with 95.4% accuracy. Includes treatment lookup
from crop_knowledge knowledge base.
"""

import json
import logging
import os
from typing import Dict, Any, Optional
from PIL import Image

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DISEASE_MODEL_HF, DISEASE_CLASSES, DATA_RAW_DIR

logger = logging.getLogger(__name__)


class DiseasePredictor:
    """
    Crop disease prediction using pre-trained HuggingFace MobileNetV2.
    No training, no GPU — just download and infer.
    """

    def __init__(self):
        self.classifier = None
        self.treatments = {}
        self._load_model()
        self._load_treatments()

    def _load_model(self):
        """Load disease classification model (MLflow → HuggingFace fallback)."""
        # Try MLflow first (Databricks deployment)
        try:
            import mlflow
            self.classifier = mlflow.transformers.load_model(
                "models:/krishimitra-disease-classifier/latest"
            )
            logger.info("✅ Disease model loaded from MLflow registry")
            return
        except Exception as e:
            logger.info(f"MLflow unavailable ({e}), loading from HuggingFace...")

        # Fallback: load directly from HuggingFace
        try:
            from transformers import pipeline
            self.classifier = pipeline(
                "image-classification",
                model=DISEASE_MODEL_HF,
                top_k=5,
            )
            logger.info(f"✅ Disease model loaded from HuggingFace: {DISEASE_MODEL_HF}")
        except Exception as e:
            logger.error(f"❌ Failed to load disease model: {e}")
            self.classifier = None

    def _load_treatments(self):
        """Load disease treatment knowledge base from crop_knowledge.json."""
        kb_path = os.path.join(DATA_RAW_DIR, "crop_knowledge.json")
        if os.path.exists(kb_path):
            try:
                with open(kb_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # Index by disease_class key
                for entry in data:
                    key = entry.get("disease_class", "")
                    if key:
                        self.treatments[key] = entry
                logger.info(f"✅ Loaded {len(self.treatments)} treatment entries from crop_knowledge.json")
            except Exception as e:
                logger.error(f"Failed to load crop_knowledge.json: {e}")
        else:
            logger.warning(f"⚠️ crop_knowledge.json not found at {kb_path}")
            self._load_default_treatments()

    def _load_default_treatments(self):
        """Fallback minimal treatment data for the 38 PlantVillage classes."""
        defaults = {
            "Apple___Apple_scab": {
                "treatment": "Apply fungicides like Captan or Mancozeb. Remove infected leaves and fruit.",
                "prevention": "Use resistant varieties. Ensure good air circulation. Rake fallen leaves.",
                "organic": "Neem oil spray. Sulfur-based fungicide.",
            },
            "Apple___Black_rot": {
                "treatment": "Prune infected branches. Apply Captan-based fungicide.",
                "prevention": "Remove mummified fruits. Maintain tree hygiene.",
                "organic": "Copper-based organic spray.",
            },
            "Apple___Cedar_apple_rust": {
                "treatment": "Apply Myclobutanil or Mancozeb during spring.",
                "prevention": "Remove nearby juniper/cedar trees if possible.",
                "organic": "Sulfur spray during early infection.",
            },
            "Tomato___Late_blight": {
                "treatment": "Apply Metalaxyl + Mancozeb (Ridomil Gold). Remove affected plants immediately.",
                "prevention": "Avoid overhead irrigation. Use drip irrigation. Space plants well.",
                "organic": "Bordeaux mixture spray. Trichoderma soil application.",
            },
            "Tomato___Early_blight": {
                "treatment": "Apply Chlorothalonil or Mancozeb fungicide. Remove lower infected leaves.",
                "prevention": "Mulch around base. Rotate crops every 2-3 years.",
                "organic": "Neem oil spray. Baking soda solution (1 tbsp per gallon).",
            },
            "Tomato___Bacterial_spot": {
                "treatment": "Copper-based bactericide spray. Remove infected plants.",
                "prevention": "Use disease-free seeds. Avoid working in wet fields.",
                "organic": "Copper hydroxide spray. Bacillus subtilis application.",
            },
            "Tomato___Leaf_Mold": {
                "treatment": "Improve ventilation. Apply Chlorothalonil fungicide.",
                "prevention": "Reduce humidity in greenhouse. Space plants adequately.",
                "organic": "Potassium bicarbonate spray. Improve airflow.",
            },
            "Tomato___Septoria_leaf_spot": {
                "treatment": "Apply Mancozeb or Chlorothalonil. Remove lower infected leaves.",
                "prevention": "Mulch to prevent soil splash. Rotate crops.",
                "organic": "Copper-based spray. Remove infected debris.",
            },
            "Tomato___Spider_mites Two-spotted_spider_mite": {
                "treatment": "Apply Dicofol or Abamectin miticide. Wash plants with strong water spray.",
                "prevention": "Maintain adequate moisture. Avoid drought stress.",
                "organic": "Neem oil spray. Release predatory mites (Phytoseiulus persimilis).",
            },
            "Tomato___Target_Spot": {
                "treatment": "Apply Mancozeb or Copper oxychloride.",
                "prevention": "Remove plant debris. Ensure good air circulation.",
                "organic": "Trichoderma viride application. Neem-based spray.",
            },
            "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
                "treatment": "No cure. Remove infected plants. Control whitefly vectors with Imidacloprid.",
                "prevention": "Use resistant varieties. Install yellow sticky traps. Use reflective mulch.",
                "organic": "Neem oil for whitefly control. Companion planting with marigold.",
            },
            "Tomato___Tomato_mosaic_virus": {
                "treatment": "No cure. Remove and destroy infected plants. Disinfect tools.",
                "prevention": "Use virus-free seeds. Wash hands before handling plants.",
                "organic": "Milk spray (1:9 ratio) can reduce spread. Remove weeds.",
            },
            "Potato___Early_blight": {
                "treatment": "Apply Mancozeb or Chlorothalonil at first sign of infection.",
                "prevention": "Crop rotation with non-Solanaceous crops. Use certified seed.",
                "organic": "Copper-based spray. Trichoderma application.",
            },
            "Potato___Late_blight": {
                "treatment": "Apply Metalaxyl + Mancozeb (Ridomil). Destroy infected tubers.",
                "prevention": "Plant resistant varieties. Avoid excessive irrigation.",
                "organic": "Bordeaux mixture. Destroy volunteer potato plants.",
            },
            "Corn_(maize)___Common_rust_": {
                "treatment": "Apply Propiconazole or Mancozeb fungicide.",
                "prevention": "Plant resistant hybrids. Early planting.",
                "organic": "Sulfur-based fungicide. Crop rotation.",
            },
            "Corn_(maize)___Northern_Leaf_Blight": {
                "treatment": "Apply Azoxystrobin or Propiconazole. Remove crop residue.",
                "prevention": "Use resistant varieties. Rotate with non-cereal crops.",
                "organic": "Trichoderma-based biocontrol. Deep plowing of residue.",
            },
            "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
                "treatment": "Apply Azoxystrobin + Propiconazole.",
                "prevention": "Tillage to bury residue. Resistant hybrids.",
                "organic": "Crop rotation. Residue management.",
            },
            "Grape___Black_rot": {
                "treatment": "Apply Mancozeb or Myclobutanil before flowering.",
                "prevention": "Remove mummified berries. Prune for air circulation.",
                "organic": "Copper-based spray. Sanitation pruning.",
            },
            "Grape___Esca_(Black_Measles)": {
                "treatment": "No effective chemical cure. Remove severely affected vines.",
                "prevention": "Avoid large pruning wounds. Protect wounds with sealant.",
                "organic": "Trichoderma application to pruning wounds.",
            },
            "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
                "treatment": "Apply Mancozeb or Carbendazim fungicide.",
                "prevention": "Good canopy management. Remove fallen leaves.",
                "organic": "Bordeaux mixture spray. Neem oil.",
            },
            "Cherry_(including_sour)___Powdery_mildew": {
                "treatment": "Apply Myclobutanil or Sulfur-based fungicide.",
                "prevention": "Prune for air circulation. Avoid excessive nitrogen.",
                "organic": "Potassium bicarbonate spray. Neem oil.",
            },
            "Peach___Bacterial_spot": {
                "treatment": "Copper-based spray during dormant season.",
                "prevention": "Plant resistant varieties. Avoid overhead irrigation.",
                "organic": "Copper hydroxide application at petal fall.",
            },
            "Pepper,_bell___Bacterial_spot": {
                "treatment": "Copper-based bactericide + Mancozeb.",
                "prevention": "Use disease-free transplants. Crop rotation.",
                "organic": "Copper spray. Bacillus subtilis application.",
            },
            "Squash___Powdery_mildew": {
                "treatment": "Apply Myclobutanil or Sulfur fungicide.",
                "prevention": "Space plants for air circulation. Avoid overhead watering.",
                "organic": "Baking soda spray. Neem oil. Milk spray (1:9).",
            },
            "Strawberry___Leaf_scorch": {
                "treatment": "Apply Captan or Thiram fungicide. Remove affected leaves.",
                "prevention": "Use resistant cultivars. Renovate beds after harvest.",
                "organic": "Copper-based spray. Remove old foliage.",
            },
            "Orange___Haunglongbing_(Citrus_greening)": {
                "treatment": "No cure. Remove infected trees. Control psyllid vectors with Imidacloprid.",
                "prevention": "Use disease-free nursery stock. Control Asian citrus psyllid.",
                "organic": "Neem oil for psyllid control. Nutritional sprays to manage symptoms.",
            },
        }

        for disease_class in DISEASE_CLASSES:
            if disease_class not in defaults and "healthy" not in disease_class.lower():
                defaults[disease_class] = {
                    "treatment": "Consult a local agriculture officer or KVK for specific treatment.",
                    "prevention": "Practice crop rotation, use certified seeds, and maintain field hygiene.",
                    "organic": "Use neem-based sprays and biological control agents.",
                }

        self.treatments = {
            k: {"disease_class": k, **v} for k, v in defaults.items()
        }
        logger.info(f"✅ Loaded {len(self.treatments)} default treatment entries")

    def predict(self, image_input) -> Dict[str, Any]:
        """
        Predict crop disease from a leaf image.

        Args:
            image_input: File path (str) or PIL Image

        Returns:
            Dict with keys: disease, disease_raw, crop, confidence,
                            is_healthy, treatment, prevention, organic_option,
                            top_predictions
        """
        if self.classifier is None:
            return {
                "error": "Disease model not loaded. Please check installation.",
                "disease": "Unknown",
                "crop": "Unknown",
                "confidence": 0.0,
                "is_healthy": False,
                "treatment": "Model unavailable. Please try again later.",
                "prevention": "",
                "organic_option": "",
                "top_predictions": [],
            }

        try:
            # Handle different input types
            if isinstance(image_input, str):
                image = Image.open(image_input).convert("RGB")
            elif isinstance(image_input, Image.Image):
                image = image_input.convert("RGB")
            else:
                image = Image.fromarray(image_input).convert("RGB")

            # Run inference
            results = self.classifier(image)
            top = results[0]

            disease_label = top["label"]  # e.g., "Tomato___Late_blight"
            parts = disease_label.split("___")
            crop = parts[0].replace("_", " ") if parts else "Unknown"
            disease = parts[1].replace("_", " ") if len(parts) > 1 else "Unknown"
            is_healthy = "healthy" in disease.lower()

            # Fetch treatment from knowledge base
            treatment_info = self.treatments.get(disease_label, {})

            return {
                "disease": disease,
                "disease_raw": disease_label,
                "crop": crop,
                "confidence": round(top["score"], 4),
                "is_healthy": is_healthy,
                "treatment": treatment_info.get("treatment", "Consult a local agriculture expert."),
                "prevention": treatment_info.get("prevention", "Practice good farm hygiene."),
                "organic_option": treatment_info.get("organic", "Use neem-based sprays."),
                "top_predictions": [
                    {"label": r["label"], "score": round(r["score"], 4)}
                    for r in results[:5]
                ],
            }

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {
                "error": str(e),
                "disease": "Error",
                "crop": "Unknown",
                "confidence": 0.0,
                "is_healthy": False,
                "treatment": f"Error during prediction: {e}",
                "prevention": "",
                "organic_option": "",
                "top_predictions": [],
            }

    def format_result(self, result: Dict[str, Any]) -> str:
        """Format prediction result into readable markdown."""
        if result.get("error"):
            return f"❌ **Error**: {result['error']}"

        if result["is_healthy"]:
            return (
                f"### ✅ Healthy Plant Detected!\n\n"
                f"**Crop**: {result['crop']}\n"
                f"**Confidence**: {result['confidence']*100:.1f}%\n\n"
                f"Your plant appears healthy! Keep following good agricultural practices."
            )

        return (
            f"### 🦠 Disease Detected: **{result['disease']}**\n\n"
            f"**Crop**: {result['crop']}\n"
            f"**Confidence**: {result['confidence']*100:.1f}%\n\n"
            f"---\n\n"
            f"#### 💊 Treatment\n{result['treatment']}\n\n"
            f"#### 🛡️ Prevention\n{result['prevention']}\n\n"
            f"#### 🌿 Organic Alternative\n{result['organic_option']}\n\n"
            f"---\n"
            f"*⚠️ This is an AI prediction (95.4% model accuracy). "
            f"Always consult a local agriculture officer for confirmation.*"
        )


# Module-level singleton
_predictor = None


def get_disease_predictor() -> DiseasePredictor:
    """Get or create the singleton DiseasePredictor instance."""
    global _predictor
    if _predictor is None:
        _predictor = DiseasePredictor()
    return _predictor
