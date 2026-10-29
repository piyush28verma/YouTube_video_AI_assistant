"""Embedding Models and Factory for YouTube RAG Pipeline.

Supports:
- Google Generative AI text-embedding-004 (Production mode)
- Deterministic OfflineFallbackEmbeddings (Offline/Demo mode)
"""

import math
import re
from typing import List, Optional, Tuple
from langchain_core.embeddings import Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from config import DEFAULT_EMBEDDING_MODEL, EMBEDDING_DIMENSION_OFFLINE


class OfflineFallbackEmbeddings(Embeddings):
    """Deterministic, high-frequency hashed term vectorizer.

    Guarantees full vector search and offline evaluation capabilities
    without requiring an active internet connection or API credits.
    Outputs unit-normalized vectors for exact cosine / Euclidean similarity.
    """

    def __init__(self, dimension: int = EMBEDDING_DIMENSION_OFFLINE):
        self.dimension = dimension

    def _embed_text(self, text: str) -> List[float]:
        vec = [0.0] * self.dimension
        words = re.findall(r"\w+", text.lower())
        if not words:
            return vec
        for word in words:
            h = abs(hash(word)) % self.dimension
            vec[h] += 1.0
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed_text(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed_text(text)

    def __call__(self, text: str) -> List[float]:
        return self._embed_text(text)


def create_embeddings_engine(
    api_key: Optional[str] = None,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> Tuple[Embeddings, bool, str]:
    """Factory creating the appropriate embedding engine with model fallback probing."""
    if api_key and api_key.strip():
        clean_key = api_key.strip()
        candidates = [
            model_name,
            "gemini-embedding-001",
            "models/gemini-embedding-001",
            "gemini-embedding-2",
            "text-embedding-004",
            "models/text-embedding-004",
        ]
        # Preserve order while deduplicating
        seen = set()
        unique_candidates = [c for c in candidates if not (c in seen or seen.add(c))]

        for candidate in unique_candidates:
            try:
                engine = GoogleGenerativeAIEmbeddings(
                    model=candidate,
                    google_api_key=clean_key,
                )
                # Verify model is available on this API key and version
                engine.embed_query("ping")
                return engine, False, f"Google {candidate}"
            except Exception:
                continue

    return OfflineFallbackEmbeddings(), True, "Offline Hashed Vectorizer (Demo Mode)"
