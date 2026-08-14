"""
POST /api/v1/pipeline/svg/ingest      – run the pipeline on SVG files already on disk
POST /api/v1/pipeline/svg/upload      – upload SVG file(s) via multipart/form-data
POST /api/v1/pipeline/svg/upload-raw  – send raw SVG content as JSON string
POST /api/v1/pipeline/svg/search      – semantic search against the Qdrant svg_icons collection
"""

from __future__ import annotations

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional

# Thread pool used to run the sync LangGraph pipeline outside FastAPI's event loop
_PIPELINE_EXECUTOR = ThreadPoolExecutor(max_workers=4)

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile, status
from pydantic import BaseModel, Field

from app.pipelines.svg_ingest import SvgIconItem, SvgPipelineState, svg_ingest_graph

router = APIRouter(prefix="/pipeline/svg", tags=["SVG Pipeline"])


# ── Schemas ────────────────────────────────────────────────────────────────────

class SvgIngestRequest(BaseModel):
    """Request body for triggering the SVG ingestion pipeline."""

    svg_dir: str = Field(
        default="svg_icon",
        description=(
            "Path to the directory containing .svg files. "
            "Relative paths are resolved from the project root."
        ),
    )
    file_list: list[str] = Field(
        default=[],
        description=(
            "Optional explicit list of filenames (e.g. ['account.svg', 'alarm.svg']). "
            "If empty, all *.svg files in svg_dir are processed."
        ),
    )
    background: bool = Field(
        default=False,
        description="If true, run pipeline in background and return immediately.",
    )


class SvgIngestResponse(BaseModel):
    message: str
    total_scanned: int
    processed_count: int
    failed_count: int
    errors: list[str]
    icons: list[dict[str, Any]] = Field(
        default=[],
        description="Summary of each processed icon (slug, name, public_url, db_id).",
    )


class SvgRawUploadRequest(BaseModel):
    """Send one or more SVG icons as raw string content in a JSON body."""
    icons: list[dict] = Field(
        ...,
        description=(
            "List of icons, each with 'filename' and 'svg_content' keys. "
            "Example: [{\"filename\": \"account.svg\", \"svg_content\": \"<svg>...</svg>\"}]"
        ),
    )


class SvgUploadResponse(BaseModel):
    message: str
    total_uploaded: int
    processed_count: int
    failed_count: int
    errors: list[str]
    icons: list[dict[str, Any]] = Field(
        default=[],
        description="Summary of each processed icon (slug, name, public_url, db_id).",
    )


class SvgSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Natural language search query")
    top_k: int = Field(5, ge=1, le=50)
    score_threshold: float = Field(0.0, ge=0.0, le=1.0)
    category: Optional[str] = Field(None, description="Filter by icon category")
    style: Optional[str] = Field(None, description="Filter by icon style (outline, filled, ...)")


class SvgSearchHit(BaseModel):
    score: float
    payload: dict[str, Any]


class SvgSearchResponse(BaseModel):
    query: str
    results: list[SvgSearchHit]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _resolve_svg_dir(svg_dir: str) -> str:
    """
    Resolve a possibly-relative svg_dir to an absolute path.
    Relative paths are anchored to the project root (cwd or CWD env var).
    """
    p = Path(svg_dir)
    if p.is_absolute():
        return str(p)
    # Try CWD first, then fall back to env variable PROJECT_ROOT
    project_root = Path(os.environ.get("PROJECT_ROOT", Path.cwd()))
    return str(project_root / p)


def _run_svg_pipeline(svg_dir: str, file_list: list[str]) -> SvgPipelineState:
    """Build initial state and invoke the LangGraph pipeline synchronously."""
    initial: SvgPipelineState = {
        "svg_dir":         svg_dir,
        "file_list":       file_list,
        "icons":           [],
        "processed_count": 0,
        "failed_count":    0,
        "errors":          [],
    }
    # The graph nodes are sync; ainvoke is not needed here but we use invoke
    return svg_ingest_graph.invoke(initial)


# ── POST /pipeline/svg/ingest ─────────────────────────────────────────────────
@router.post(
    "/ingest",
    response_model=SvgIngestResponse,
    status_code=status.HTTP_200_OK,
    summary="Run the SVG icon ingestion pipeline",
    description=(
        "Scans a directory for .svg files, enriches each one with Gemini LLM "
        "(generates name, description, tags, category, style, use_cases), "
        "uploads to Cloudflare R2, persists metadata to PostgreSQL, "
        "embeds descriptions with Gemini embedding-001, "
        "and upserts vectors into the Qdrant 'svg_icons' collection."
    ),
)
async def ingest_svg_icons(
    body: SvgIngestRequest,
    background_tasks: BackgroundTasks,
) -> SvgIngestResponse:
    svg_dir = _resolve_svg_dir(body.svg_dir)

    if body.background:
        # Fire and forget
        background_tasks.add_task(_run_svg_pipeline, svg_dir, body.file_list)
        return SvgIngestResponse(
            message="SVG ingestion pipeline queued in background.",
            total_scanned=0,
            processed_count=0,
            failed_count=0,
            errors=[],
            icons=[],
        )

    try:
        loop = asyncio.get_event_loop()
        final: SvgPipelineState = await loop.run_in_executor(
            _PIPELINE_EXECUTOR,
            _run_svg_pipeline,
            svg_dir,
            body.file_list,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"SVG pipeline error: {exc}",
        )

    icons_summary = [
        {
            "slug":       i.get("slug"),
            "name":       i.get("name"),
            "category":   i.get("category"),
            "style":      i.get("style"),
            "public_url": i.get("public_url"),
            "db_id":      i.get("db_id"),
            "error":      i.get("error"),
        }
        for i in final.get("icons", [])
    ]

    total_scanned = len(final.get("icons", []))
    processed = final.get("processed_count", 0)
    failed = final.get("failed_count", 0)

    return SvgIngestResponse(
        message=f"SVG pipeline complete. {processed}/{total_scanned} icons processed.",
        total_scanned=total_scanned,
        processed_count=processed,
        failed_count=failed,
        errors=final.get("errors", []),
        icons=icons_summary,
    )


# ── POST /pipeline/svg/upload ────────────────────────────────────────────────────
@router.post(
    "/upload",
    response_model=SvgUploadResponse,
    status_code=status.HTTP_200_OK,
    summary="Upload SVG file(s) and run the full ingestion pipeline",
    description=(
        "Accepts one or more SVG files via multipart/form-data. "
        "Each file is passed directly through the pipeline: "
        "LLM enrichment (name, description, tags, category, style) → "
        "Cloudflare R2 upload → PostgreSQL save → "
        "Gemini embedding → Qdrant upsert. "
        "No files need to exist on the server's disk."
    ),
)
async def upload_svg_icons(
    files: list[UploadFile] = File(..., description="One or more .svg files"),
) -> SvgUploadResponse:
    # ── Validate & read all uploaded files ───────────────────────────────────
    pre_icons: list[SvgIconItem] = []
    errors: list[str] = []

    for upload in files:
        filename = upload.filename or "unknown.icon"
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        if ext == "jpeg":
            ext = "jpg"

        allowed_exts = {"svg", "png", "jpg", "webp", "ico", "gif"}
        if ext not in allowed_exts:
            errors.append(f"{filename}: unsupported format '{ext}' (allowed: svg, png, jpg, webp, ico, gif)")
            continue

        raw = await upload.read()
        if not raw:
            errors.append(f"{filename}: empty file, skipped")
            continue

        mime_type = upload.content_type or f"image/{ext}"
        if ext == "ico" and "icon" not in mime_type:
            mime_type = "image/x-icon"

        svg_content = None
        if ext == "svg":
            try:
                svg_content = raw.decode("utf-8")
            except UnicodeDecodeError:
                errors.append(f"{filename}: could not decode SVG as UTF-8, skipped")
                continue

        slug = filename.rsplit(".", 1)[0].lower().replace(" ", "-")
        pre_icons.append(
            SvgIconItem(
                filename=filename,
                slug=slug,
                file_format=ext,
                svg_content=svg_content,
                raw_bytes=raw,
                mime_type=mime_type,
                file_size_bytes=len(raw),
                error=None,
            )
        )

    if not pre_icons:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"No valid icon files provided. Errors: {errors}",
        )

    def _run_upload_pipeline(icons: list, errs: list) -> SvgPipelineState:
        """Run the upload pipeline in a thread (so asyncio.run() works inside nodes)."""
        from langgraph.graph import StateGraph, END
        from app.pipelines.svg_ingest import (
            llm_enrich, upload_r2, save_db, embed_and_upsert, done,
            SvgPipelineState,
        )

        def _skip_scan(state: SvgPipelineState) -> SvgPipelineState:
            return state

        g = StateGraph(SvgPipelineState)
        g.add_node("skip_scan",       _skip_scan)
        g.add_node("llm_enrich",      llm_enrich)
        g.add_node("upload_r2",        upload_r2)
        g.add_node("save_db",          save_db)
        g.add_node("embed_and_upsert", embed_and_upsert)
        g.add_node("done",             done)
        g.set_entry_point("skip_scan")
        g.add_edge("skip_scan",       "llm_enrich")
        g.add_edge("llm_enrich",      "upload_r2")
        g.add_edge("upload_r2",       "save_db")
        g.add_edge("save_db",         "embed_and_upsert")
        g.add_edge("embed_and_upsert","done")
        g.add_edge("done",            END)
        compiled = g.compile()

        init: SvgPipelineState = {
            "svg_dir": "", "file_list": [],
            "icons": icons, "processed_count": 0,
            "failed_count": 0, "errors": errs,
        }
        return compiled.invoke(init)

    try:
        loop = asyncio.get_event_loop()
        final: SvgPipelineState = await loop.run_in_executor(
            _PIPELINE_EXECUTOR,
            _run_upload_pipeline,
            pre_icons,
            errors,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"SVG upload pipeline error: {exc}",
        )

    icons_summary = [
        {
            "slug":       i.get("slug"),
            "name":       i.get("name"),
            "category":   i.get("category"),
            "style":      i.get("style"),
            "tags":       i.get("tags", []),
            "public_url": i.get("public_url"),
            "db_id":      i.get("db_id"),
            "error":      i.get("error"),
        }
        for i in final.get("icons", [])
    ]

    total = len(final.get("icons", []))
    processed = final.get("processed_count", 0)
    failed = final.get("failed_count", 0)

    return SvgUploadResponse(
        message=f"Upload pipeline complete. {processed}/{total} icons processed.",
        total_uploaded=total,
        processed_count=processed,
        failed_count=failed,
        errors=final.get("errors", []),
        icons=icons_summary,
    )


# ── POST /pipeline/svg/upload-raw ────────────────────────────────────────────
@router.post(
    "/upload-raw",
    response_model=SvgUploadResponse,
    status_code=status.HTTP_200_OK,
    summary="Send raw SVG content as JSON and run the full ingestion pipeline",
    description=(
        "Accepts a JSON body with a list of icons, each having a 'filename' and "
        "'svg_content' string. Ideal for sending SVG markup directly from code "
        "without multipart encoding. Runs the full pipeline: LLM enrichment → "
        "R2 upload → PostgreSQL save → Gemini embedding → Qdrant upsert."
    ),
)
async def upload_svg_raw(
    body: SvgRawUploadRequest,
) -> SvgUploadResponse:
    pre_icons: list[SvgIconItem] = []
    errors: list[str] = []

    for item in body.icons:
        filename = str(item.get("filename") or "unknown.svg")
        svg_content = str(item.get("svg_content") or "")

        if not svg_content.strip():
            errors.append(f"{filename}: empty svg_content, skipped")
            continue
        if "<svg" not in svg_content.lower():
            errors.append(f"{filename}: does not look like SVG markup, skipped")
            continue

        slug = filename.rsplit(".", 1)[0].lower().replace(" ", "-")
        pre_icons.append(
            SvgIconItem(
                filename=filename if filename.endswith(".svg") else filename + ".svg",
                slug=slug,
                svg_content=svg_content,
                file_size_bytes=len(svg_content.encode("utf-8")),
                error=None,
            )
        )

    if not pre_icons:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"No valid SVG content provided. Errors: {errors}",
        )

    def _run_raw_pipeline(icons: list, errs: list) -> SvgPipelineState:
        from langgraph.graph import StateGraph, END
        from app.pipelines.svg_ingest import (
            llm_enrich, upload_r2, save_db, embed_and_upsert, done,
            SvgPipelineState,
        )

        def _skip_scan(state: SvgPipelineState) -> SvgPipelineState:
            return state

        g = StateGraph(SvgPipelineState)
        g.add_node("skip_scan",       _skip_scan)
        g.add_node("llm_enrich",      llm_enrich)
        g.add_node("upload_r2",        upload_r2)
        g.add_node("save_db",          save_db)
        g.add_node("embed_and_upsert", embed_and_upsert)
        g.add_node("done",             done)
        g.set_entry_point("skip_scan")
        g.add_edge("skip_scan",       "llm_enrich")
        g.add_edge("llm_enrich",      "upload_r2")
        g.add_edge("upload_r2",       "save_db")
        g.add_edge("save_db",         "embed_and_upsert")
        g.add_edge("embed_and_upsert","done")
        g.add_edge("done",            END)
        compiled = g.compile()

        init: SvgPipelineState = {
            "svg_dir": "", "file_list": [],
            "icons": icons, "processed_count": 0,
            "failed_count": 0, "errors": errs,
        }
        return compiled.invoke(init)

    try:
        loop = asyncio.get_event_loop()
        final: SvgPipelineState = await loop.run_in_executor(
            _PIPELINE_EXECUTOR,
            _run_raw_pipeline,
            pre_icons,
            errors,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"SVG raw upload pipeline error: {exc}",
        )

    icons_summary = [
        {
            "slug":       i.get("slug"),
            "name":       i.get("name"),
            "category":   i.get("category"),
            "style":      i.get("style"),
            "tags":       i.get("tags", []),
            "public_url": i.get("public_url"),
            "db_id":      i.get("db_id"),
            "error":      i.get("error"),
        }
        for i in final.get("icons", [])
    ]
    total = len(final.get("icons", []))
    processed = final.get("processed_count", 0)
    failed = final.get("failed_count", 0)

    return SvgUploadResponse(
        message=f"Raw upload pipeline complete. {processed}/{total} icons processed.",
        total_uploaded=total,
        processed_count=processed,
        failed_count=failed,
        errors=final.get("errors", []),
        icons=icons_summary,
    )


# ── POST /pipeline/svg/search ─────────────────────────────────────────────────
@router.post(
    "/search",
    response_model=SvgSearchResponse,
    summary="Semantic search across SVG icons",
    description=(
        "Embeds the query with Gemini embedding-001 and performs cosine "
        "similarity search in the Qdrant 'svg_icons' collection."
    ),
)
async def search_svg_icons(body: SvgSearchRequest) -> SvgSearchResponse:
    try:
        from app.pipelines.svg_ingest import SVG_COLLECTION, _embed_texts
        from app.services.qdrant_service import get_qdrant_client
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        vectors = _embed_texts([body.query])
        query_vector = vectors[0]

        # Build optional payload filter
        must_conditions = []
        if body.category:
            must_conditions.append(
                FieldCondition(key="category", match=MatchValue(value=body.category))
            )
        if body.style:
            must_conditions.append(
                FieldCondition(key="style", match=MatchValue(value=body.style))
            )

        qdrant_filter = Filter(must=must_conditions) if must_conditions else None

        client = get_qdrant_client()
        response = client.query_points(
            collection_name=SVG_COLLECTION,
            query=query_vector,
            limit=body.top_k,
            score_threshold=body.score_threshold,
            query_filter=qdrant_filter,
            with_payload=True,
        )
        results = response.points
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"SVG search failed: {exc}",
        )

    return SvgSearchResponse(
        query=body.query,
        results=[
            SvgSearchHit(score=hit.score, payload=hit.payload or {})
            for hit in results
        ],
    )
