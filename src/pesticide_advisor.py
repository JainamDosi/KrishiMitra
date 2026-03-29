"""
KrishiMitra — Pesticide & Fertilizer Advisor (RAG)
=====================================================
Retrieval-Augmented Generation for crop-specific pesticide and
fertilizer recommendations. Uses FAISS vector search + Sarvam AI.
"""

import json
import logging
import os
from typing import Dict, Any, List, Optional

import numpy as np
import requests

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    DATA_RAW_DIR, MODELS_DIR,
    PESTICIDE_FAISS_INDEX, PESTICIDE_CHUNKS_PATH,
    SARVAM_API_KEY, SARVAM_CHAT_ENDPOINT, SARVAM_CHAT_MODEL,
)

logger = logging.getLogger(__name__)


class PesticideAdvisor:
    """
    RAG-based pesticide & fertilizer recommendation engine.
    Pipeline: Crop + Stage + Problem → FAISS Search → Context → Sarvam AI → Advice
    """

    def __init__(self):
        self.index = None
        self.embed_model = None
        self.chunks = []
        self.guide_data = []
        self._load_guide()
        self._load_index()

    def _load_guide(self):
        """Load pesticide/fertilizer guide from JSON."""
        json_path = os.path.join(DATA_RAW_DIR, "pesticide_fertilizer.json")
        if os.path.exists(json_path):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    self.guide_data = json.load(f)
                logger.info(f"✅ Loaded {len(self.guide_data)} pesticide/fertilizer entries")
                self._build_chunks()
            except Exception as e:
                logger.error(f"Failed to load pesticide_fertilizer.json: {e}")
        else:
            logger.warning(f"⚠️ pesticide_fertilizer.json not found at {json_path}")

    def _build_chunks(self):
        """Create searchable text chunks from guide data."""
        self.chunks = []
        for entry in self.guide_data:
            crop = entry.get("crop", "")
            stage = entry.get("growth_stage", "")
            category = entry.get("category", "")
            problem = entry.get("problem", "")

            text = (
                f"Crop: {crop}\n"
                f"Growth Stage: {stage}\n"
                f"Category: {category}\n"
                f"Problem: {problem}\n"
                f"Product: {entry.get('product_name', '')}\n"
                f"Dosage: {entry.get('dosage', '')}\n"
                f"Application Method: {entry.get('application_method', '')}\n"
                f"Timing: {entry.get('timing', '')}\n"
                f"Precautions: {entry.get('precautions', '')}\n"
                f"Organic Alternative: {entry.get('organic_alternative', '')}\n"
                f"Cost Estimate: {entry.get('cost_estimate', '')}"
            )

            self.chunks.append({
                "crop": crop,
                "growth_stage": stage,
                "category": category,
                "text": text,
            })

        logger.info(f"✅ Built {len(self.chunks)} pesticide/fertilizer text chunks")

    def _load_index(self):
        """Load or build FAISS index."""
        if os.path.exists(PESTICIDE_FAISS_INDEX) and os.path.exists(PESTICIDE_CHUNKS_PATH):
            try:
                import faiss
                self.index = faiss.read_index(PESTICIDE_FAISS_INDEX)
                with open(PESTICIDE_CHUNKS_PATH, "r", encoding="utf-8") as f:
                    self.chunks = json.load(f)
                self._load_embed_model()
                logger.info("✅ Loaded pre-built pesticide FAISS index")
                return
            except Exception as e:
                logger.warning(f"Failed to load pre-built index: {e}")

        if self.chunks:
            self._build_index()

    def _load_embed_model(self):
        """Load sentence transformer model."""
        if self.embed_model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
            self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("✅ Loaded sentence transformer model")
        except Exception as e:
            logger.error(f"Failed to load sentence transformer: {e}")

    def _build_index(self):
        """Build FAISS index from pesticide chunks."""
        if not self.chunks:
            return

        self._load_embed_model()
        if self.embed_model is None:
            return

        try:
            import faiss

            texts = [c["text"] for c in self.chunks]
            embeddings = self.embed_model.encode(texts, show_progress_bar=False)
            embeddings_np = np.array(embeddings).astype("float32")
            faiss.normalize_L2(embeddings_np)

            self.index = faiss.IndexFlatIP(embeddings_np.shape[1])
            self.index.add(embeddings_np)

            # Save for reuse
            os.makedirs(MODELS_DIR, exist_ok=True)
            faiss.write_index(self.index, PESTICIDE_FAISS_INDEX)
            with open(PESTICIDE_CHUNKS_PATH, "w", encoding="utf-8") as f:
                json.dump(self.chunks, f, ensure_ascii=False, indent=2)

            logger.info(f"✅ Built FAISS index with {len(self.chunks)} pesticide chunks")
        except Exception as e:
            logger.error(f"Failed to build FAISS index: {e}")

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant pesticide/fertilizer information."""
        if self.index is None or self.embed_model is None:
            return self._keyword_search(query, top_k)

        try:
            q_emb = self.embed_model.encode(query).reshape(1, -1).astype("float32")
            import faiss
            faiss.normalize_L2(q_emb)
            scores, indices = self.index.search(q_emb, top_k)

            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.chunks):
                    chunk = self.chunks[idx].copy()
                    chunk["score"] = float(score)
                    results.append(chunk)

            return results
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return self._keyword_search(query, top_k)

    def _keyword_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Fallback keyword-based search."""
        query_lower = query.lower()
        scored = []
        for chunk in self.chunks:
            text = chunk["text"].lower()
            score = sum(1 for word in query_lower.split() if word in text)
            if score > 0:
                chunk_copy = chunk.copy()
                chunk_copy["score"] = score
                scored.append(chunk_copy)

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def get_recommendation(self, crop: str, stage: str = "",
                            problem: str = "", prefer_organic: bool = False,
                            language: str = "en") -> str:
        """
        Get pesticide/fertilizer recommendation using RAG.

        Args:
            crop: Crop name (e.g., "Rice", "Wheat")
            stage: Growth stage (e.g., "Tillering", "Flowering")
            problem: Specific problem description
            prefer_organic: Whether to prefer organic solutions
            language: User's language code

        Returns:
            Recommendation text in the user's language
        """
        # Build search query
        query_parts = [crop]
        if stage:
            query_parts.append(stage)
        if problem:
            query_parts.append(problem)
        if prefer_organic:
            query_parts.append("organic biological natural")

        query = " ".join(query_parts)

        # Translate if needed
        from src.translator import get_translator
        translator = get_translator()

        en_query = translator.translate(query, language, "en") if language != "en" else query

        # Search
        results = self.search(en_query, top_k=5)

        if not results:
            no_result = (
                f"No specific recommendation found for {crop}"
                f"{' at ' + stage + ' stage' if stage else ''}. "
                f"Please consult your nearest Krishi Vigyan Kendra (KVK) or "
                f"agriculture department for personalized advice."
            )
            return translator.translate(no_result, "en", language) if language != "en" else no_result

        context = "\n\n---\n\n".join([r["text"] for r in results])

        # Generate answer using Sarvam AI
        answer = self._generate_recommendation(en_query, context, prefer_organic)

        # Translate back
        if language != "en":
            answer = translator.translate(answer, "en", language)

        return answer

    def _generate_recommendation(self, query: str, context: str,
                                   prefer_organic: bool = False) -> str:
        """Generate recommendation using Sarvam AI."""
        if not SARVAM_API_KEY:
            return f"Based on available guidance:\n\n{context}"

        organic_note = (
            "\nThe farmer prefers ORGANIC solutions. Prioritize organic alternatives, "
            "bio-pesticides, and natural remedies. Only mention chemical options as last resort."
            if prefer_organic else ""
        )

        system_prompt = f"""You are KrishiMitra, an expert crop protection advisor for Indian farmers.
Your role is to provide safe, accurate pesticide and fertilizer recommendations.

Rules:
1. Answer based ONLY on the provided product information
2. Always include: Product name, Dosage, Application method, Timing
3. Always include safety precautions — farmer safety is paramount
4. Include organic alternatives when available
5. Include approximate cost when known
6. Use simple, farmer-friendly language
7. Warn about waiting periods before harvest
8. Recommend protective equipment{organic_note}"""

        user_prompt = f"""Based on this pesticide/fertilizer guide:

{context}

---

Farmer's Query: {query}

Provide a specific, actionable recommendation with clear dosage and application instructions."""

        try:
            headers = {
                "Content-Type": "application/json",
                "API-Subscription-Key": SARVAM_API_KEY,
            }
            payload = {
                "model": SARVAM_CHAT_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.3,
                "max_tokens": 1000,
            }

            response = requests.post(
                SARVAM_CHAT_ENDPOINT,
                json=payload,
                headers=headers,
                timeout=30,
            )
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]

        except Exception as e:
            logger.error(f"Sarvam AI chat error: {e}")
            return f"Based on available guidance:\n\n{context}"

    def get_crops(self) -> List[str]:
        """Get list of crops covered in the guide."""
        crops = set()
        for entry in self.guide_data:
            crop = entry.get("crop", "")
            if crop:
                crops.add(crop)
        return sorted(crops)

    def get_stages(self, crop: str) -> List[str]:
        """Get growth stages for a specific crop."""
        stages = set()
        for entry in self.guide_data:
            if entry.get("crop", "") == crop:
                stage = entry.get("growth_stage", "")
                if stage:
                    stages.add(stage)
        return sorted(stages)

    def get_categories(self) -> List[str]:
        """Get available product categories."""
        return sorted(set(
            entry.get("category", "") for entry in self.guide_data if entry.get("category")
        ))


# Module-level singleton
_advisor = None


def get_pesticide_advisor() -> PesticideAdvisor:
    """Get or create the singleton PesticideAdvisor instance."""
    global _advisor
    if _advisor is None:
        _advisor = PesticideAdvisor()
    return _advisor
