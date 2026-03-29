"""
KrishiMitra — Chat Engine & Intent Router
===========================================
Classifies user intent and routes to appropriate feature module:
 - Disease detection
 - Price prediction
 - Government scheme advice
 - Pesticide/fertilizer recommendation
"""

import logging
import re
from typing import Dict, Any, Tuple

import requests
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SARVAM_API_KEY, SARVAM_CHAT_ENDPOINT, SARVAM_CHAT_MODEL

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Intent Classification
# ──────────────────────────────────────────────

class IntentClassifier:
    """
    Classifies farmer queries into one of four feature intents.
    Uses keyword-based classification with Sarvam AI fallback.
    """

    INTENTS = {
        "disease": {
            "keywords": [
                "disease", "blight", "rust", "rot", "scab", "mold", "mildew",
                "spots", "yellow", "brown", "wilt", "infection", "fungus",
                "leaves", "leaf", "photo", "image", "upload", "scan", "detect",
                "sick", "infected", "dying", "पत्ती", "बीमारी", "रोग", "कीट",
                "rog", "bimari", "patti", "keeda",
            ],
            "description": "Crop disease detection from leaf images",
        },
        "price": {
            "keywords": [
                "price", "rate", "mandi", "market", "sell", "buy", "cost",
                "forecast", "predict", "trend", "dal", "quintal",
                "wholesale", "retail", "monsoon", "season",
                "कीमत", "दाम", "मंडी", "बाजार", "भाव", "बेचना",
                "kimat", "daam", "bazaar", "bhav", "bechna",
            ],
            "description": "Mandi price analysis & forecasting",
        },
        "scheme": {
            "keywords": [
                "scheme", "yojana", "subsidy", "loan", "credit", "insurance",
                "pm-kisan", "pmfby", "kcc", "e-nam", "pmkisan", "government",
                "benefit", "apply", "registration", "eligibility",
                "sarkari", "sarkar", "labh", "avedan",
                "योजना", "सब्सिडी", "लाभ", "सरकार", "ऋण", "बीमा", "आवेदन",
            ],
            "description": "Government scheme advice",
        },
        "pesticide": {
            "keywords": [
                "pesticide", "fertilizer", "spray", "urea", "dap", "npk",
                "insecticide", "herbicide", "fungicide", "weedicide",
                "dose", "dosage", "organic", "neem", "bio", "chemical",
                "nitrogen", "phosphorus", "potassium", "zinc",
                "application", "when to apply", "how to apply",
                "कीटनाशक", "खाद", "उर्वरक", "दवा", "छिड़काव",
                "dawai", "khad", "urvarak", "chhidkav", "kitanashak",
            ],
            "description": "Pesticide & fertilizer recommendation",
        },
    }

    def classify(self, query: str) -> Tuple[str, float]:
        """
        Classify user query into intent.

        Returns:
            Tuple of (intent_name, confidence_score)
        """
        query_lower = query.lower().strip()

        if not query_lower:
            return "general", 0.0

        # Score each intent by keyword matches
        scores = {}
        for intent, config in self.INTENTS.items():
            score = sum(
                1 for kw in config["keywords"]
                if kw.lower() in query_lower
            )
            scores[intent] = score

        max_intent = max(scores, key=scores.get)
        max_score = scores[max_intent]

        if max_score == 0:
            # No keywords matched — try Sarvam AI classification
            return self._classify_with_llm(query)

        # Normalize confidence
        total = sum(scores.values())
        confidence = max_score / total if total > 0 else 0.0

        return max_intent, min(confidence, 1.0)

    def _classify_with_llm(self, query: str) -> Tuple[str, float]:
        """Use Sarvam AI to classify ambiguous queries."""
        if not SARVAM_API_KEY:
            return "general", 0.3

        try:
            headers = {
                "Content-Type": "application/json",
                "API-Subscription-Key": SARVAM_API_KEY,
            }
            payload = {
                "model": SARVAM_CHAT_MODEL,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Classify this farmer's question into EXACTLY ONE category. "
                            "Respond with ONLY one word:\n"
                            "- disease (plant disease/infection)\n"
                            "- price (market/mandi prices)\n"
                            "- scheme (government scheme/subsidy)\n"
                            "- pesticide (fertilizer/pesticide advice)\n"
                            "- general (other)"
                        ),
                    },
                    {"role": "user", "content": query},
                ],
                "temperature": 0.1,
                "max_tokens": 10,
            }

            response = requests.post(
                SARVAM_CHAT_ENDPOINT,
                json=payload,
                headers=headers,
                timeout=10,
            )
            response.raise_for_status()
            result = response.json()
            answer = result["choices"][0]["message"]["content"].strip().lower()

            # Parse response
            for intent in self.INTENTS:
                if intent in answer:
                    return intent, 0.7

            return "general", 0.5

        except Exception as e:
            logger.error(f"LLM classification failed: {e}")
            return "general", 0.3


# ──────────────────────────────────────────────
# Chat Engine
# ──────────────────────────────────────────────

class ChatEngine:
    """
    Main conversational engine that routes queries to feature modules.
    """

    def __init__(self):
        self.classifier = IntentClassifier()
        self.history = []

    def process_query(self, query: str, language: str = "en") -> Dict[str, Any]:
        """
        Process a user query: classify intent → route → respond.

        Args:
            query: User's message
            language: Language code

        Returns:
            Dict with keys: intent, confidence, response, feature
        """
        intent, confidence = self.classifier.classify(query)

        response = {
            "intent": intent,
            "confidence": round(confidence, 2),
            "query": query,
            "language": language,
        }

        if intent == "disease":
            response["response"] = (
                "🦠 I can help detect crop diseases! Please upload a photo of "
                "the affected leaf in the **Disease Prediction** tab, and I'll "
                "identify the disease with treatment recommendations."
            )
            response["feature"] = "disease_tab"

        elif intent == "price":
            response["response"] = (
                "📊 I can show you mandi prices and forecasts! Head to the "
                "**Price Prediction** tab, select your commodity and state, "
                "and I'll show you current prices, trends, and predictions."
            )
            response["feature"] = "price_tab"
            # Try to extract commodity from query
            response["extracted"] = self._extract_commodity(query)

        elif intent == "scheme":
            try:
                from src.scheme_advisor import get_scheme_advisor
                advisor = get_scheme_advisor()
                answer = advisor.answer_query(query, language)
                response["response"] = answer
                response["feature"] = "scheme_tab"
            except Exception as e:
                logger.error(f"Scheme advisor error: {e}")
                response["response"] = (
                    "🏛️ I can help with government scheme information! "
                    "Please use the **Scheme Knowledge** tab for detailed advice."
                )

        elif intent == "pesticide":
            try:
                from src.pesticide_advisor import get_pesticide_advisor
                advisor = get_pesticide_advisor()
                # Try to extract crop from query
                crop = self._extract_crop(query)
                answer = advisor.get_recommendation(
                    crop=crop, problem=query, language=language
                )
                response["response"] = answer
                response["feature"] = "pesticide_tab"
            except Exception as e:
                logger.error(f"Pesticide advisor error: {e}")
                response["response"] = (
                    "🧪 I can recommend pesticides and fertilizers! "
                    "Use the **Pesticide & Fertilizer** tab for detailed recommendations."
                )

        else:
            response["response"] = self._handle_general(query, language)
            response["feature"] = "general"

        # Log to history
        self.history.append({
            "query": query,
            "intent": intent,
            "response_preview": response["response"][:100],
        })

        return response

    def _extract_commodity(self, query: str) -> Dict[str, str]:
        """Extract commodity, state, market from a price query."""
        commodities = [
            "wheat", "rice", "paddy", "tomato", "onion", "potato",
            "soybean", "cotton", "maize", "chilli", "gram", "mustard",
            "banana", "mango", "sugarcane", "groundnut", "jowar",
            "bajra", "ragi", "tur", "urad", "moong", "masoor",
        ]
        query_lower = query.lower()
        found = {}
        for c in commodities:
            if c in query_lower:
                found["commodity"] = c.title()
                break
        return found

    def _extract_crop(self, query: str) -> str:
        """Extract crop name from a query."""
        crops = [
            "rice", "wheat", "tomato", "potato", "cotton", "soybean",
            "maize", "corn", "onion", "sugarcane", "grape", "chilli",
            "pepper", "apple", "banana", "mango",
        ]
        query_lower = query.lower()
        for crop in crops:
            if crop in query_lower:
                return crop.title()
        return "General"

    def _handle_general(self, query: str, language: str = "en") -> str:
        """Handle general/unclassified queries."""
        if not SARVAM_API_KEY:
            return (
                "🌾 **Namaste! I'm KrishiMitra — your AI farming assistant.**\n\n"
                "I can help you with:\n"
                "1. 🦠 **Disease Detection** — Upload a leaf photo\n"
                "2. 📊 **Price Prediction** — Check mandi prices & forecasts\n"
                "3. 🏛️ **Scheme Knowledge** — Government scheme information\n"
                "4. 🧪 **Pesticide & Fertilizer** — Get dosage recommendations\n\n"
                "Please use the tabs above or ask me a specific question!"
            )

        try:
            headers = {
                "Content-Type": "application/json",
                "API-Subscription-Key": SARVAM_API_KEY,
            }
            payload = {
                "model": SARVAM_CHAT_MODEL,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are KrishiMitra, a friendly AI farming assistant for Indian farmers. "
                            "You help with: crop disease detection, mandi price prediction, "
                            "government scheme advice, and pesticide/fertilizer recommendations. "
                            "Keep responses brief, helpful, and farmer-friendly."
                        ),
                    },
                    {"role": "user", "content": query},
                ],
                "temperature": 0.5,
                "max_tokens": 500,
            }

            response = requests.post(
                SARVAM_CHAT_ENDPOINT,
                json=payload,
                headers=headers,
                timeout=15,
            )
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]

        except Exception as e:
            logger.error(f"General chat error: {e}")
            return (
                "🌾 **Namaste! I'm KrishiMitra.**\n\n"
                "I can help with disease detection, price prediction, "
                "government schemes, and pesticide advice. "
                "Please use the tabs above!"
            )


# Module-level singleton
_engine = None


def get_chat_engine() -> ChatEngine:
    """Get or create the singleton ChatEngine instance."""
    global _engine
    if _engine is None:
        _engine = ChatEngine()
    return _engine
