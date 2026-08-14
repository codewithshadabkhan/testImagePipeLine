"""
POST /api/v1/pipeline/search   – Unified RAG search across all asset collections.

Flow
────
1. User sends a natural-language query
2. Query is embedded with Gemini (HuggingFace fallback)
3. Semantic search runs against the chosen collection(s)
4. Top-k results are fed to Gemini as context
5. Gemini generates a rich, contextual answer grounded in the retrieved assets
"""

from __future__ import annotations

import logging
from typing import Any, Literal, Optional

import google.genai as genai
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from qdrant_client.models import FieldCondition, Filter, MatchValue

from app.core.config import get_settings
from app.services.embedding_service import embed_texts_with_fallback
from app.services.qdrant_service import get_qdrant_client

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter(prefix="/pipeline", tags=["Pipeline Search"])

# ── Qdrant collection names ───────────────────────────────────────────────────
COLLECTIONS = {
    "images": settings.QDRANT_COLLECTION_NAME,   # image_assets
    "icons":  "svg_icons",                        # svg_icons / raster icons
    "all":    None,                               # search both
}

LLM_MODEL = "models/gemini-2.0-flash"


# ── Request / Response schemas ─────────────────────────────────────────────────

class PipelineSearchRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Natural-language search query from the user",
        examples=["Show me icons for navigation or maps"],
    )
    collection: Literal["images", "icons", "all"] = Field(
        default="all",
        description="Which collection to search: 'images', 'icons', or 'all'",
    )
    top_k: int = Field(5, ge=1, le=20, description="Number of results to retrieve per collection")
    score_threshold: float = Field(0.0, ge=0.0, le=1.0)
    # Optional payload filters
    category: Optional[str] = Field(None, description="Filter by category (e.g. 'ui', 'nature')")
    style: Optional[str] = Field(None, description="Filter by icon style (e.g. 'outline', 'filled') — icons only")
    # Whether to generate an LLM answer on top of results
    generate_answer: bool = Field(
        default=True,
        description="If true, Gemini generates a contextual answer from the retrieved results",
    )


class SearchHit(BaseModel):
    collection: str
    score: float
    name: Optional[str] = None
    description: Optional[str] = None
    public_url: Optional[str] = None
    category: Optional[str] = None
    style: Optional[str] = None
    tags: list[str] = []
    payload: dict[str, Any] = {}


class PipelineSearchResponse(BaseModel):
    query: str
    collection_searched: str
    total_hits: int
    hits: list[SearchHit]
    generated_answer: Optional[str] = None
    embedding_provider: str = "gemini"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_filter(category: Optional[str], style: Optional[str]) -> Optional[Filter]:
    conditions = []
    if category:
        conditions.append(FieldCondition(key="category", match=MatchValue(value=category)))
    if style:
        conditions.append(FieldCondition(key="style", match=MatchValue(value=style)))
    return Filter(must=conditions) if conditions else None


def _search_collection(
    collection_name: str,
    query_vector: list[float],
    top_k: int,
    score_threshold: float,
    qdrant_filter: Optional[Filter],
    collection_label: str,
) -> list[SearchHit]:
    """Run query_points against one Qdrant collection and return SearchHit list."""
    client = get_qdrant_client()
    try:
        response = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=top_k,
            score_threshold=score_threshold,
            query_filter=qdrant_filter,
            with_payload=True,
        )
    except Exception as exc:
        logger.warning(f"Search in '{collection_name}' failed: {exc}")
        return []

    hits = []
    for pt in response.points:
        p = pt.payload or {}
        hits.append(
            SearchHit(
                collection=collection_label,
                score=round(pt.score, 4),
                name=p.get("name") or p.get("title"),
                description=p.get("description", ""),
                public_url=p.get("public_url"),
                category=p.get("category"),
                style=p.get("style"),
                tags=p.get("tags", []),
                payload=p,
            )
        )
    return hits


def _generate_answer(query: str, hits: list[SearchHit]) -> str:
    """Call Gemini to generate a contextual answer grounded in retrieved assets."""
    if not hits:
        return "No matching assets were found for your query."

    # Build a compact context block from retrieved hits
    context_lines = []
    for i, hit in enumerate(hits, 1):
        line = f"{i}. [{hit.collection}] {hit.name or 'Unnamed'}"
        if hit.description:
            line += f": {hit.description[:200]}"
        if hit.tags:
            line += f" | Tags: {', '.join(hit.tags[:5])}"
        if hit.public_url:
            line += f" | URL: {hit.public_url}"
        context_lines.append(line)

    context = "\n".join(context_lines)

    system_prompt = (
        "You are a helpful assistant for an asset management system. "
        "The user searched for assets and got the following results. "
        "Generate a helpful, concise response that:\n"
        "1. Summarizes what was found\n"
        "2. Highlights the most relevant results with their names and URLs\n"
        "3. Explains why each result is relevant to the user's query\n"
        "Keep the answer under 250 words. Be direct and actionable."
    )
    user_message = f"User query: {query}\n\nRetrieved assets:\n{context}"

    try:
        client = genai.Client(api_key=settings.GEMINI_API_KEY)
        response = client.models.generate_content(
            model=LLM_MODEL,
            contents=[{"role": "user", "parts": [{"text": user_message}]}],
            config={"system_instruction": system_prompt, "temperature": 0.3},
        )
        return response.text.strip()
    except Exception as exc:
        logger.warning(f"LLM answer generation failed: {exc}")
        # Fallback: structured plain-text summary
        lines = [f"Found {len(hits)} result(s) for '{query}':\n"]
        for hit in hits:
            lines.append(f"• {hit.name or 'Asset'} ({hit.collection}) — score {hit.score}")
            if hit.public_url:
                lines.append(f"  URL: {hit.public_url}")
        return "\n".join(lines)


# ── POST /pipeline/search ──────────────────────────────────────────────────────

@router.post(
    "/search",
    response_model=PipelineSearchResponse,
    status_code=status.HTTP_200_OK,
    summary="Unified semantic search across all asset collections",
    description=(
        "Takes a natural-language query from the user, embeds it with Gemini "
        "(HuggingFace open-source fallback on error), searches the chosen "
        "Qdrant collection(s), and optionally generates a contextual AI answer "
        "grounded in the retrieved results (RAG-style)."
    ),
)
async def pipeline_search(body: PipelineSearchRequest) -> PipelineSearchResponse:

    # ── 1. Embed the query ─────────────────────────────────────────────────────
    try:
        vectors, embed_provider = embed_texts_with_fallback([body.query])
        query_vector = vectors[0]
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Embedding failed: {exc}",
        )

    # ── 2. Build optional filter ───────────────────────────────────────────────
    qdrant_filter = _build_filter(body.category, body.style)

    # ── 3. Search collection(s) ────────────────────────────────────────────────
    all_hits: list[SearchHit] = []

    if body.collection in ("images", "all"):
        all_hits += _search_collection(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            query_vector=query_vector,
            top_k=body.top_k,
            score_threshold=body.score_threshold,
            qdrant_filter=qdrant_filter,
            collection_label="images",
        )

    if body.collection in ("icons", "all"):
        all_hits += _search_collection(
            collection_name="svg_icons",
            query_vector=query_vector,
            top_k=body.top_k,
            score_threshold=body.score_threshold,
            qdrant_filter=qdrant_filter,
            collection_label="icons",
        )

    # Sort combined results by score descending
    all_hits.sort(key=lambda h: h.score, reverse=True)

    # ── 4. Generate LLM answer (optional) ─────────────────────────────────────
    generated_answer: Optional[str] = None
    if body.generate_answer:
        generated_answer = _generate_answer(body.query, all_hits)

    return PipelineSearchResponse(
        query=body.query,
        collection_searched=body.collection,
        total_hits=len(all_hits),
        hits=all_hits,
        generated_answer=generated_answer,
        embedding_provider=embed_provider,
    )
