"""
KrishiMitra — Sarvam AI Translation Wrapper
=============================================
Provides translate() for converting between 10+ Indian languages
using Sarvam AI's translation API.
"""

import requests
import logging
from typing import Optional

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SARVAM_API_KEY, SARVAM_TRANSLATE_ENDPOINT, SARVAM_BASE_URL

logger = logging.getLogger(__name__)


class Translator:
    """Sarvam AI translation wrapper for Indian languages."""

    # Sarvam supported language codes
    SUPPORTED_LANGS = {
        "en", "hi", "ta", "te", "kn", "ml", "mr", "bn", "gu", "pa", "or"
    }

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or SARVAM_API_KEY
        self.translate_url = SARVAM_TRANSLATE_ENDPOINT
        if not self.api_key:
            logger.warning("⚠️ No Sarvam API key provided. Translation will passthrough.")

    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        """
        Translate text between Indian languages using Sarvam AI.

        Args:
            text: Input text to translate
            source_lang: Source language code (e.g., 'hi', 'en', 'ta')
            target_lang: Target language code

        Returns:
            Translated text, or original text on failure
        """
        # No-op if same language
        if source_lang == target_lang:
            return text

        # Validate languages
        if source_lang not in self.SUPPORTED_LANGS:
            logger.warning(f"Unsupported source language: {source_lang}")
            return text
        if target_lang not in self.SUPPORTED_LANGS:
            logger.warning(f"Unsupported target language: {target_lang}")
            return text

        # Skip if no API key
        if not self.api_key:
            logger.warning("No API key — returning original text")
            return text

        try:
            headers = {
                "Content-Type": "application/json",
                "API-Subscription-Key": self.api_key,
            }
            payload = {
                "input": text,
                "source_language_code": source_lang,
                "target_language_code": target_lang,
                "mode": "formal",
                "model": "mayura:v1",
                "enable_preprocessing": True,
            }

            response = requests.post(
                self.translate_url,
                json=payload,
                headers=headers,
                timeout=15,
            )
            response.raise_for_status()
            result = response.json()

            translated = result.get("translated_text", text)
            logger.info(
                f"Translated [{source_lang}→{target_lang}]: "
                f"'{text[:50]}...' → '{translated[:50]}...'"
            )
            return translated

        except requests.exceptions.Timeout:
            logger.error("Translation API timed out")
            return text
        except requests.exceptions.RequestException as e:
            logger.error(f"Translation API error: {e}")
            return text
        except Exception as e:
            logger.error(f"Unexpected translation error: {e}")
            return text

    def detect_language(self, text: str) -> str:
        """
        Detect the language of input text using Sarvam AI.
        Falls back to 'en' if detection fails.
        """
        if not self.api_key:
            return "en"

        try:
            headers = {
                "Content-Type": "application/json",
                "API-Subscription-Key": self.api_key,
            }
            payload = {"input": text}

            response = requests.post(
                f"{SARVAM_BASE_URL}/detect-language",
                json=payload,
                headers=headers,
                timeout=10,
            )
            response.raise_for_status()
            result = response.json()
            return result.get("language_code", "en")

        except Exception as e:
            logger.error(f"Language detection failed: {e}")
            return "en"


# Module-level singleton
_translator = None


def get_translator() -> Translator:
    """Get or create the singleton Translator instance."""
    global _translator
    if _translator is None:
        _translator = Translator()
    return _translator
