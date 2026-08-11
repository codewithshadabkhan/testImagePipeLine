"""
POST /api/v1/pipeline/vectorize   – run LangGraph pipeline (fetch DB → embed → upsert Qdrant)
POST /api/v1/pipeline/search      – semantic similarity search against Qdrant
"""

from datetime import datetime
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.pipelines.vectorize import PipelineState, vectorize_graph
from app.services.image_service import get_all_image_assets
from app.services.search_service import semantic_search

router = APIRouter(prefix="/pipeline", tags=["Pipeline"])


# ── Schemas ────────────────────────────────────────────────────────────────────
class VectorizeResponse(BaseModel):
    message: str
    total_fetched: int
    upserted_count: int
    skipped: int
    errors: list[str]


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Natural language search query")
    top_k: int = Field(5, ge=1, le=50, description="Number of results to return")
    score_threshold: float = Field(0.0, ge=0.0, le=1.0, description="Minimum cosine similarity score")

    # ── Payload filters (each hits a pre-built Qdrant index → fast) ──────────────
    category: Optional[str] = Field(
        None,
        description="Filter by category keyword (nature, technology, architecture, people, art, other)"
    )
    image_type: Optional[str] = Field(
        None,
        description="Filter by image type keyword (jpeg, png, webp, gif, avif)"
    )
    min_size: Optional[int] = Field(None, ge=0, description="Min file size in bytes")
    max_size: Optional[int] = Field(None, ge=0, description="Max file size in bytes")
    min_width: Optional[int] = Field(None, ge=1, description="Min image width in pixels")
    min_height: Optional[int] = Field(None, ge=1, description="Min image height in pixels")
    created_after: Optional[datetime] = Field(None, description="Filter images created after this datetime (ISO 8601)")
    created_before: Optional[datetime] = Field(None, description="Filter images created before this datetime (ISO 8601)")


class SearchHit(BaseModel):
    score: float
    payload: dict[str, Any]


class SearchResponse(BaseModel):
    query: str
    results: list[SearchHit]


# ── POST /pipeline/vectorize ───────────────────────────────────────────────────
@router.post(
    "/vectorize",
    response_model=VectorizeResponse,
    summary="Vectorize all DB images into Qdrant",
    description=(
        "Fetches every image record from PostgreSQL, embeds the description "
        "with Google Gemini text-embedding-004, and upserts vectors + full metadata "
        "into the Qdrant collection. Safe to run multiple times (upsert is idempotent)."
    ),
)
async def run_vectorize_pipeline(
    db: AsyncSession = Depends(get_db),
) -> VectorizeResponse:
    # 1. Fetch all records from Postgres
    total, assets = await get_all_image_assets(db, limit=10_000)

    if not assets:
        return VectorizeResponse(
            message="No image records found in the database.",
            total_fetched=0,
            upserted_count=0,
            skipped=0,
            errors=[],
        )

    # 2. Serialize ORM objects to plain dicts for the pipeline
    records = [
        {
            "id":             asset.id,
            "title":          asset.title,
            "description":    asset.description,
            "category":       str(asset.category),
            "image_type":     str(asset.image_type),
            "public_url":     asset.public_url,
            "r2_key":         asset.r2_key,
            "file_size_bytes":asset.file_size_bytes,
            "width":          asset.width,
            "height":         asset.height,
            "mime_type":      asset.mime_type,
            "created_at":     asset.created_at,
        }
        for asset in assets
    ]

    # 3. Run the LangGraph pipeline
    try:
        initial_state: PipelineState = {
            "records":       records,
            "embedded":      [],
            "upserted_count":0,
            "errors":        [],
        }
        final_state: PipelineState = await vectorize_graph.ainvoke(initial_state)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pipeline error: {exc}",
        )

    skipped = total - final_state["upserted_count"] - len(
        [e for e in final_state["errors"] if "Skipped" in e]
    )

    return VectorizeResponse(
        message=f"Pipeline complete. {final_state['upserted_count']} vectors upserted.",
        total_fetched=total,
        upserted_count=final_state["upserted_count"],
        skipped=max(0, total - final_state["upserted_count"]),
        errors=final_state["errors"],
    )


# ── POST /pipeline/search ──────────────────────────────────────────────────────
@router.post(
    "/search",
    response_model=SearchResponse,
    summary="Semantic image search",
    description=(
        "Embeds the query text with Gemini and returns the most semantically "
        "similar image records from Qdrant, ranked by cosine similarity score."
    ),
)
async def search_images(body: SearchRequest) -> SearchResponse:
    try:
        hits = semantic_search(
            query=body.query,
            top_k=body.top_k,
            score_threshold=body.score_threshold,
            # ── pass indexed filters ───────────────────────────────────────
            category=body.category,
            image_type=body.image_type,
            min_size=body.min_size,
            max_size=body.max_size,
            min_width=body.min_width,
            min_height=body.min_height,
            created_after=body.created_after,
            created_before=body.created_before,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Search failed: {exc}",
        )

    return SearchResponse(
        query=body.query,
        results=[SearchHit(**h) for h in hits],
    )
