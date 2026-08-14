"""
Embedding service with primary Gemini API and automatic HuggingFace fallback.

Primary: Google Gemini (models/gemini-embedding-001) - 3072 dimensions
Fallback: Hugging Face Inference API (sentence-transformers/all-MiniLM-L6-v2)
"""

from __future__ import annotations

import json
import logging
import urllib.request
import google.genai as genai

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

GEMINI_EMBED_MODEL = "models/gemini-embedding-001"


def embed_texts_hf(texts: list[str]) -> list[list[float]]:
    """
    Fallback embedding generator using HuggingFace Inference API.
    Uses urllib to avoid heavy external dependencies.
    """
    model_name = settings.HF_EMBEDDING_MODEL or "sentence-transformers/all-MiniLM-L6-v2"
    url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_name}"
    
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "FastAPI-Pipeline/1.0",
    }
    if settings.HF_TOKEN:
        headers["Authorization"] = f"Bearer {settings.HF_TOKEN}"

    payload = json.dumps({"inputs": texts, "options": {"wait_for_model": True}}).encode("utf-8")
    req = urllib.request.Request(url, data=payload, headers=headers, method="POST")

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            # Hugging Face feature extraction returns list of vectors (or list of token embeddings)
            if isinstance(data, list) and len(data) > 0:
                # If shape is [batch, tokens, hidden], mean pool across tokens
                if isinstance(data[0], list) and len(data[0]) > 0 and isinstance(data[0][0], list):
                    vectors = []
                    for doc in data:
                        # Mean pooling over token embeddings
                        dim = len(doc[0])
                        mean_vec = [sum(token[i] for token in doc) / len(doc) for i in range(dim)]
                        vectors.append(mean_vec)
                    return vectors
                # If shape is [batch, hidden]
                elif isinstance(data[0], list) and isinstance(data[0][0], (int, float)):
                    return data
            raise ValueError(f"Unexpected HuggingFace API output structure: {data[:1]}")
    except Exception as exc:
        logger.error(f"HuggingFace embedding fallback failed: {exc}")
        raise RuntimeError(f"Hugging Face embedding failed: {exc}") from exc


def embed_texts_with_fallback(texts: list[str]) -> tuple[list[list[float]], str]:
    """
    Embeds a list of strings.
    Tries Google Gemini first. If Gemini fails (overloaded/503/rate limit/error),
    falls back to HuggingFace Inference API open-source model.

    Returns:
        (vectors, provider_used)
    """
    if not texts:
        return [], "none"

    # Attempt 1: Gemini Primary
    try:
        client = genai.Client(api_key=settings.GEMINI_API_KEY)
        result = client.models.embed_content(
            model=GEMINI_EMBED_MODEL,
            contents=texts,
        )
        vectors = [e.values for e in result.embeddings]
        return vectors, "gemini"
    except Exception as exc:
        logger.warning(f"Primary Gemini embedding failed ({exc}). Triggering HuggingFace open-source model fallback...")

    # Attempt 2: HuggingFace Open-Source Fallback
    try:
        vectors = embed_texts_hf(texts)
        return vectors, "huggingface"
    except Exception as exc:
        logger.error(f"Both Gemini and HuggingFace fallbacks failed: {exc}")
        raise RuntimeError(f"All embedding providers failed. Gemini err; HF err: {exc}") from exc
