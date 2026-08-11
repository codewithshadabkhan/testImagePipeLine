from datetime import datetime
from typing import Any, Optional

import google.genai as genai
from qdrant_client.models import (
    DatetimeRange,
    FieldCondition,
    Filter,
    MatchValue,
    Range,
)

from app.core.config import get_settings
from app.services.qdrant_service import get_qdrant_client

settings = get_settings()

EMBED_MODEL = "models/gemini-embedding-001"


def embed_query(text: str) -> list[float]:
    """
    Embed a single query string using Gemini text-embedding-004 via v1 API.
    Uses the google-genai SDK directly to avoid the langchain v1beta issue.
    """
    client = genai.Client(api_key=settings.GEMINI_API_KEY)
    result = client.models.embed_content(
        model=EMBED_MODEL,
        contents=[text],
    )
    return result.embeddings[0].values


def _build_filter(
    category: Optional[str] = None,
    image_type: Optional[str] = None,
    min_size: Optional[int] = None,
    max_size: Optional[int] = None,
    min_width: Optional[int] = None,
    min_height: Optional[int] = None,
    created_after: Optional[datetime] = None,
    created_before: Optional[datetime] = None,
) -> Optional[Filter]:
    """
    Build a Qdrant Filter from optional criteria.

    Each condition hits a pre-built payload index:
      category / image_type  → KEYWORD index  (exact match, O(log n))
      file_size_bytes         → INTEGER index  (range query)
      width / height          → INTEGER index  (range query)
      created_at              → DATETIME index (range query)
    """
    conditions = []

    # ── keyword filters ────────────────────────────────────────────────────────
    if category:
        conditions.append(
            FieldCondition(key="category", match=MatchValue(value=category))
        )
    if image_type:
        conditions.append(
            FieldCondition(key="image_type", match=MatchValue(value=image_type))
        )

    # ── integer range filters ──────────────────────────────────────────────────
    if min_size is not None or max_size is not None:
        conditions.append(
            FieldCondition(
                key="file_size_bytes",
                range=Range(gte=min_size, lte=max_size),
            )
        )
    if min_width is not None:
        conditions.append(
            FieldCondition(key="width", range=Range(gte=min_width))
        )
    if min_height is not None:
        conditions.append(
            FieldCondition(key="height", range=Range(gte=min_height))
        )

    # ── datetime range filter ──────────────────────────────────────────────────
    if created_after or created_before:
        conditions.append(
            FieldCondition(
                key="created_at",
                range=DatetimeRange(
                    gte=created_after.isoformat() if created_after else None,
                    lte=created_before.isoformat() if created_before else None,
                ),
            )
        )

    if not conditions:
        return None

    return Filter(must=conditions)


def semantic_search(
    query: str,
    top_k: int = 5,
    score_threshold: float = 0.0,
    # ── payload filters (use indexed fields for speed) ─────────────────────────
    category: Optional[str] = None,
    image_type: Optional[str] = None,
    min_size: Optional[int] = None,
    max_size: Optional[int] = None,
    min_width: Optional[int] = None,
    min_height: Optional[int] = None,
    created_after: Optional[datetime] = None,
    created_before: Optional[datetime] = None,
) -> list[dict[str, Any]]:
    """
    Embed a query string and return the top-k closest image records.

    Filtered search uses Qdrant's HNSW + payload index combination:
      1. HNSW graph narrows down candidates by vector similarity
      2. Payload index instantly filters by category / type / size / date
      This is much faster than post-filtering on raw results.

    Returns: [ { score: float, payload: { db_id, title, public_url, ... } } ]
    """
    embedder = embed_query
    query_vector = embedder(query)

    query_filter = _build_filter(
        category=category,
        image_type=image_type,
        min_size=min_size,
        max_size=max_size,
        min_width=min_width,
        min_height=min_height,
        created_after=created_after,
        created_before=created_before,
    )

    client = get_qdrant_client()
    results = client.search(
        collection_name=settings.QDRANT_COLLECTION_NAME,
        query_vector=query_vector,
        query_filter=query_filter,      # uses payload indexes automatically
        limit=top_k,
        score_threshold=score_threshold,
        with_payload=True,
    )

    return [
        {
            "score":   round(hit.score, 4),
            "payload": hit.payload,
        }
        for hit in results
    ]
