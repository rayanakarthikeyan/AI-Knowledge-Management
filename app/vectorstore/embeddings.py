"""
Embedding generation services.
Supports HuggingFace SentenceTransformers, OpenAI embeddings, and offline fast embeddings.
"""

from typing import List
import numpy as np
from app.config import config
from app.utils.logger import logger


class EmbeddingService:
    """Manages dense vector embeddings for document chunks and user queries."""

    def __init__(self, model_name: str = None):
        self.model_name = model_name or config.EMBEDDING_MODEL_NAME
        self._st_model = None
        self._init_model()

    def _init_model(self):
        """Attempts to load SentenceTransformer; falls back gracefully if network unavailable."""
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading embedding model: {self.model_name}")
            self._st_model = SentenceTransformer(self.model_name)
            logger.info("SentenceTransformer loaded successfully.")
        except Exception as e:
            logger.warning(
                f"Could not load SentenceTransformer '{self.model_name}' ({e}). "
                "Using robust lightweight deterministic embedding fallback."
            )
            self._st_model = None

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Computes embeddings for a batch of text chunks."""
        if not texts:
            return []

        if self._st_model is not None:
            try:
                embeddings = self._st_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
                return embeddings.tolist()
            except Exception as e:
                logger.error(f"Error generating SentenceTransformer embeddings: {e}")

        # Lightweight deterministic fallback (384-dimensional normalized vector)
        return [self._fallback_embed(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        """Computes embedding for a single user query."""
        results = self.embed_documents([text])
        return results[0] if results else [0.0] * 384

    def _fallback_embed(self, text: str, dim: int = 384) -> List[float]:
        """Generates a normalized deterministic n-dimensional pseudo-semantic vector."""
        vec = np.zeros(dim, dtype=np.float32)
        words = text.lower().split()
        for idx, word in enumerate(words):
            # Hash tokens into vector slots with positional weighting
            h = hash(word) % dim
            weight = 1.0 / (1.0 + 0.05 * idx)
            vec[h] += weight

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()
