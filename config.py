"""Configuration and Hyperparameters for LLM-Powered YouTube Video Assistant (RAG).

Centralizes all pipeline hyperparameters, model identifiers, retrieval thresholds,
and user interface themes.
"""

import os
from dataclasses import dataclass
from typing import List


# ── Application Metadata ──
APP_NAME = "LLM-Powered YouTube Video Assistant (RAG)"
APP_SUBTITLE = "Grounded Transcript Intelligence & Real-Time Semantic Video Search"
VERSION = "2.5.0"
ICON = "🎬"


# ── Default Model Identifiers ──
DEFAULT_EMBEDDING_MODEL = "gemini-embedding-001"
DEFAULT_CHAT_MODEL = "gemini-3.8-flash"
SUPPORTED_CHAT_MODELS: List[str] = [
    "gemini-3.8-flash",
    "gemini-flash-lite-latest",
    "gemini-pro-latest",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
]


# ── RAG Pipeline Hyperparameters ──
DEFAULT_CHUNK_SIZE: int = 800
DEFAULT_CHUNK_OVERLAP: int = 150
DEFAULT_TOP_K: int = 4
DEFAULT_SEARCH_TYPE: str = "similarity"  # "similarity" or "mmr"
DEFAULT_SCORE_THRESHOLD: float = 0.35
DEFAULT_TEMPERATURE: float = 0.2
EMBEDDING_DIMENSION_OFFLINE: int = 256


# ── Retrieval Confidence Thresholds (FAISS L2 Normalized) ──
HIGH_CONFIDENCE_THRESHOLD: float = 75.0
MEDIUM_CONFIDENCE_THRESHOLD: float = 50.0


# ── Supported Subtitle Languages ──
DEFAULT_PREFERRED_LANGUAGES: List[str] = [
    "en",
    "en-US",
    "en-GB",
    "es",
    "fr",
    "de",
    "hi",
]


@dataclass
class PipelineConfig:
    """Dataclass encapsulating complete runtime parameters for RAG execution."""

    api_key: str = ""
    model_name: str = DEFAULT_CHAT_MODEL
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    top_k: int = DEFAULT_TOP_K
    search_type: str = DEFAULT_SEARCH_TYPE
    score_threshold: float = DEFAULT_SCORE_THRESHOLD
    temperature: float = DEFAULT_TEMPERATURE

    @classmethod
    def from_env(cls) -> "PipelineConfig":
        """Instantiate configuration directly from environment variables."""
        return cls(
            api_key=(os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or "").strip(),
            model_name=os.getenv("GEMINI_MODEL", DEFAULT_CHAT_MODEL),
            chunk_size=int(os.getenv("RAG_CHUNK_SIZE", str(DEFAULT_CHUNK_SIZE))),
            chunk_overlap=int(os.getenv("RAG_CHUNK_OVERLAP", str(DEFAULT_CHUNK_OVERLAP))),
            top_k=int(os.getenv("RAG_TOP_K", str(DEFAULT_TOP_K))),
            score_threshold=float(os.getenv("RAG_SCORE_THRESHOLD", str(DEFAULT_SCORE_THRESHOLD))),
            temperature=float(os.getenv("RAG_TEMPERATURE", str(DEFAULT_TEMPERATURE))),
        )
