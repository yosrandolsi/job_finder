"""
Local embedding similarity via sentence-transformers (no API key, no cost).

The model is loaded once and cached at module level -- loading
all-MiniLM-L6-v2 takes a couple of seconds, so we don't want to pay that
cost per job posting evaluated.
"""

from __future__ import annotations

from sentence_transformers import SentenceTransformer, util

_MODEL_NAME = "all-MiniLM-L6-v2"
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


def cosine_similarity(text_a: str, text_b: str) -> float:
    """Return a 0-1 cosine similarity between two texts' embeddings."""
    model = _get_model()
    embeddings = model.encode([text_a, text_b], convert_to_tensor=True, normalize_embeddings=True)
    score = util.cos_sim(embeddings[0], embeddings[1]).item()
    return max(0.0, min(1.0, float(score)))
