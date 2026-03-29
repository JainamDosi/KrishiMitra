"""
KrishiMitra — Government Scheme Advisor (RAG)
===============================================
Retrieval-Augmented Generation for Indian agricultural scheme advice.
Uses FAISS vector search + Sarvam AI for multilingual scheme recommendations.
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
    SCHEME_FAISS_INDEX, SCHEME_CHUNKS_PATH,
    SARVAM_API_KEY, SARVAM_CHAT_ENDPOINT, SARVAM_CHAT_MODEL,
)

logger = logging.getLogger(__name__)


class SchemeAdvisor:
    """
    RAG-based government scheme advisor.
    Pipeline: Query → FAISS Search → Context → Sarvam AI → Answer
    """

    def __init__(self):
        self.index = None
        self.embed_model = None
        self.chunks = []
        self.schemes_data = []
        self._load_schemes()
        self._load_index()

    def _load_schemes(self):
        """Load government schemes data from JSON."""
        json_path = os.path.join(DATA_RAW_DIR, "govt_schemes.json")
        if os.path.exists(json_path):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    self.schemes_data = json.load(f)
                logger.info(f"✅ Loaded {len(self.schemes_data)} government schemes")
                self._build_chunks()
            except Exception as e:
                logger.error(f"Failed to load govt_schemes.json: {e}")
        else:
            logger.warning(f"⚠️ govt_schemes.json not found at {json_path}")

    def _build_chunks(self):
        """Create searchable text chunks from scheme data."""
        self.chunks = []
        for scheme in self.schemes_data:
            scheme_id = scheme.get("scheme_id", "")
            name = scheme.get("name_en", scheme.get("name", ""))

            # Chunk 1: Overview + Benefits
            overview_text = (
                f"Scheme: {name}\n"
                f"Category: {scheme.get('category', '')}\n"
                f"Description: {scheme.get('description', '')}\n"
                f"Benefits: {scheme.get('benefits', '')}\n"
                f"Coverage: {scheme.get('coverage', '')}"
            )
            self.chunks.append({
                "scheme_id": scheme_id,
                "name": name,
                "chunk_type": "overview",
                "text": overview_text,
            })

            # Chunk 2: Eligibility + Application Process
            eligibility = scheme.get("eligibility", [])
            if isinstance(eligibility, list):
                eligibility = "; ".join(eligibility)
            documents = scheme.get("documents_required", [])
            if isinstance(documents, list):
                documents = ", ".join(documents)

            apply_text = (
                f"Scheme: {name}\n"
                f"Eligibility: {eligibility}\n"
                f"How to Apply: {scheme.get('how_to_apply', '')}\n"
                f"Documents Required: {documents}\n"
                f"Helpline: {scheme.get('helpline', '')}\n"
                f"Official URL: {scheme.get('official_url', '')}"
            )
            self.chunks.append({
                "scheme_id": scheme_id,
                "name": name,
                "chunk_type": "application",
                "text": apply_text,
            })

        logger.info(f"✅ Built {len(self.chunks)} scheme text chunks")

    def _load_index(self):
        """Load or build FAISS index for scheme search."""
        # Try loading pre-built index
        if os.path.exists(SCHEME_FAISS_INDEX) and os.path.exists(SCHEME_CHUNKS_PATH):
            try:
                import faiss
                self.index = faiss.read_index(SCHEME_FAISS_INDEX)
                with open(SCHEME_CHUNKS_PATH, "r", encoding="utf-8") as f:
                    self.chunks = json.load(f)
                self._load_embed_model()
                logger.info("✅ Loaded pre-built scheme FAISS index")
                return
            except Exception as e:
                logger.warning(f"Failed to load pre-built index: {e}")

        # Build index if we have chunks
        if self.chunks:
            self._build_index()

    def _load_embed_model(self):
        """Load sentence transformer for embedding generation."""
        if self.embed_model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
            self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("✅ Loaded sentence transformer model")
        except Exception as e:
            logger.error(f"Failed to load sentence transformer: {e}")

    def _build_index(self):
        """Build FAISS index from scheme chunks."""
        if not self.chunks:
            logger.warning("No chunks to index")
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

            # Save for future use
            os.makedirs(MODELS_DIR, exist_ok=True)
            faiss.write_index(self.index, SCHEME_FAISS_INDEX)
            with open(SCHEME_CHUNKS_PATH, "w", encoding="utf-8") as f:
                json.dump(self.chunks, f, ensure_ascii=False, indent=2)

            logger.info(f"✅ Built FAISS index with {len(self.chunks)} scheme chunks")
        except Exception as e:
            logger.error(f"Failed to build FAISS index: {e}")

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for relevant scheme information using vector similarity.

        Args:
            query: User's question about schemes
            top_k: Number of results to return

        Returns:
            List of matching chunks with scores
        """
        if self.index is None or self.embed_model is None:
            # Fallback: keyword search
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

    def answer_query(self, query: str, language: str = "en") -> str:
        """
        Full RAG pipeline: translate → search → generate → translate back.

        Args:
            query: User's question (may be in any Indian language)
            language: User's language code

        Returns:
            Answer string in the user's language
        """
        # Step 1: Translate to English if needed
        from src.translator import get_translator
        translator = get_translator()

        en_query = translator.translate(query, language, "en") if language != "en" else query

        # Step 2: Vector search for relevant context
        results = self.search(en_query, top_k=5)

        if not results:
            no_result = (
                "I couldn't find specific scheme information for your question. "
                "Please try asking about a specific scheme like PM-KISAN, PMFBY, "
                "KCC, e-NAM, or Soil Health Card."
            )
            return translator.translate(no_result, "en", language) if language != "en" else no_result

        context = "\n\n---\n\n".join([r["text"] for r in results])

        # Step 3: Generate answer using Sarvam AI
        answer = self._generate_answer(en_query, context)

        # Step 4: Translate back if needed
        if language != "en":
            answer = translator.translate(answer, "en", language)

        return answer

    def _generate_answer(self, query: str, context: str) -> str:
        """Generate a comprehensive answer using Sarvam AI chat API."""
        if not SARVAM_API_KEY:
            # Fallback: return context directly
            return f"Based on available information:\n\n{context}"

        system_prompt = """You are KrishiMitra, an expert agricultural advisor for Indian farmers.
Your role is to provide accurate, helpful information about government schemes for farmers.

Rules:
1. Answer based ONLY on the provided scheme information
2. Include specific details: amounts, eligibility, application steps
3. If the scheme information is incomplete, say so honestly
4. Use simple, farmer-friendly language
5. Include helpline numbers and URLs when available
6. Format with bullet points and clear structure"""

        user_prompt = f"""Based on this scheme information:

{context}

---

Farmer's Question: {query}

Please provide a detailed, helpful answer with specific actionable steps."""

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
            return f"Based on available information:\n\n{context}"

    def list_schemes(self) -> List[Dict[str, str]]:
        """Get a summary list of all available schemes."""
        return [
            {
                "scheme_id": s.get("scheme_id", ""),
                "name": s.get("name_en", s.get("name", "")),
                "short_name": s.get("name", ""),
                "category": s.get("category", ""),
                "benefits_summary": s.get("benefits", "")[:100] + "...",
            }
            for s in self.schemes_data
        ]

    def get_scheme_detail(self, scheme_id: str) -> Optional[Dict[str, Any]]:
        """Get full details for a specific scheme."""
        for scheme in self.schemes_data:
            if scheme.get("scheme_id") == scheme_id:
                return scheme
        return None


# Module-level singleton
_advisor = None


def get_scheme_advisor() -> SchemeAdvisor:
    """Get or create the singleton SchemeAdvisor instance."""
    global _advisor
    if _advisor is None:
        _advisor = SchemeAdvisor()
    return _advisor
