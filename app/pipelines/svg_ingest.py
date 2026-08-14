"""
SVG Icon Ingestion Pipeline — LangGraph implementation.

Graph nodes
───────────
1. scan_files      – read SVG files from disk (or accept a pre-loaded list)
2. llm_enrich      – call Gemini LLM to generate payload (name, desc, tags, etc.)
3. upload_r2       – upload SVG bytes to Cloudflare R2, get r2_key + public_url
4. save_db         – persist SvgIcon record to PostgreSQL (sync wrapper)
5. embed_and_upsert – embed description with Gemini embedding-001, upsert to Qdrant
6. done            – terminal summary node

State
─────
SvgPipelineState is a TypedDict flowing through every node.
Each item in `icons` carries the SVG file data + progressively enriched fields.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypedDict

import google.genai as genai
from langgraph.graph import END, StateGraph
from qdrant_client.models import PointStruct
from sqlalchemy import select

from app.core.config import get_settings
from app.db.session import async_session_factory
from app.models.svg_icon import IconCategory, IconStyle, SvgIcon
from app.services.qdrant_service import ensure_collection, get_qdrant_client
from app.services.r2_service import upload_image_to_r2
from app.services.svg_llm_service import (
    generate_svg_metadata,
    generate_icon_metadata_from_image,
)

settings = get_settings()

from app.services.embedding_service import embed_texts_with_fallback


def _embed_texts(texts: list[str]) -> list[list[float]]:
    """Batch-embed a list of strings with Gemini + HuggingFace open-source model fallback."""
    vectors, _ = embed_texts_with_fallback(texts)
    return vectors


# ── Qdrant collection for SVG icons ───────────────────────────────────────────
SVG_COLLECTION = "svg_icons"
EMBEDDING_DIM = 3072


def _ensure_svg_collection() -> None:
    """Create the SVG icon collection in Qdrant if it doesn't exist."""
    from qdrant_client.models import Distance, VectorParams, PayloadSchemaType

    client = get_qdrant_client()
    existing = {c.name for c in client.get_collections().collections}
    if SVG_COLLECTION not in existing:
        client.create_collection(
            collection_name=SVG_COLLECTION,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )
    # Create payload indexes for filterable fields
    indexes = [
        ("category",   PayloadSchemaType.KEYWORD),
        ("style",      PayloadSchemaType.KEYWORD),
        ("public_url", PayloadSchemaType.KEYWORD),
        ("slug",       PayloadSchemaType.KEYWORD),
        ("name",       PayloadSchemaType.TEXT),
        ("description",PayloadSchemaType.TEXT),
    ]
    for field, schema in indexes:
        client.create_payload_index(
            collection_name=SVG_COLLECTION,
            field_name=field,
            field_schema=schema,
        )


# ── Pipeline state ─────────────────────────────────────────────────────────────
class SvgIconItem(TypedDict, total=False):
    """Represents one icon (SVG or raster) as it flows through the pipeline."""
    filename: str               # e.g. 'account-alert.svg' or 'logo.png'
    slug: str                   # e.g. 'account-alert'
    file_format: str            # 'svg', 'png', 'jpg', 'webp', 'ico', 'gif'
    svg_content: str | None     # raw SVG markup (only for SVG)
    raw_bytes: bytes            # file bytes
    mime_type: str              # e.g. 'image/svg+xml', 'image/png'
    file_size_bytes: int

    # LLM-generated
    name: str
    description: str
    category: str
    style: str
    tags: list[str]
    use_cases: list[str]
    keywords: list[str]
    viewbox: str | None
    llm_raw: dict

    # R2
    r2_key: str
    public_url: str

    # DB
    db_id: str

    # Qdrant
    vector: list[float]
    vectorized: bool

    error: str | None           # set if this icon failed at any stage


class SvgPipelineState(TypedDict):
    svg_dir: str                        # directory containing .svg files
    file_list: list[str]                # explicit list of filenames (optional)
    icons: list[SvgIconItem]            # grows through the pipeline
    processed_count: int
    failed_count: int
    errors: list[str]


# ── Node 1 : scan_files ────────────────────────────────────────────────────────
def scan_files(state: SvgPipelineState) -> SvgPipelineState:
    """
    Read icon files from disk (supports .svg, .png, .jpg, .jpeg, .webp, .ico, .gif).
    """
    errors: list[str] = list(state.get("errors", []))
    icons: list[SvgIconItem] = []

    svg_dir = Path(state.get("svg_dir", ""))
    file_list: list[str] = state.get("file_list", [])

    allowed_exts = {".svg", ".png", ".jpg", ".jpeg", ".webp", ".ico", ".gif"}

    if file_list:
        paths = [svg_dir / fn for fn in file_list]
    else:
        if not svg_dir.is_dir():
            errors.append(f"scan_files: directory not found: {svg_dir}")
            return {**state, "icons": [], "errors": errors}
        paths = sorted([p for p in svg_dir.glob("*.*") if p.suffix.lower() in allowed_exts])

    for path in paths:
        if not path.exists():
            errors.append(f"scan_files: file not found: {path}")
            continue
        try:
            ext = path.suffix.lower().lstrip(".")
            if ext == "jpeg":
                ext = "jpg"
            
            raw = path.read_bytes()
            svg_content = None
            if ext == "svg":
                svg_content = raw.decode("utf-8", errors="ignore")
                mime = "image/svg+xml"
            else:
                mime = f"image/{ext}" if ext != "ico" else "image/x-icon"

            slug = path.stem.lower().replace(" ", "-")
            icons.append(
                SvgIconItem(
                    filename=path.name,
                    slug=slug,
                    file_format=ext,
                    svg_content=svg_content,
                    raw_bytes=raw,
                    mime_type=mime,
                    file_size_bytes=len(raw),
                    error=None,
                )
            )
        except Exception as exc:
            errors.append(f"scan_files: error reading {path.name}: {exc}")

    return {**state, "icons": icons, "errors": errors}


# ── Node 2 : llm_enrich ────────────────────────────────────────────────────────
def llm_enrich(state: SvgPipelineState) -> SvgPipelineState:
    """
    For each icon, call Gemini LLM (text for SVG, vision for raster) to generate structured metadata.
    """
    enriched: list[SvgIconItem] = []
    errors: list[str] = list(state.get("errors", []))

    for icon in state["icons"]:
        if icon.get("error"):
            enriched.append(icon)
            continue
        try:
            fmt = icon.get("file_format", "svg")
            if fmt == "svg" and icon.get("svg_content"):
                meta = generate_svg_metadata(icon["svg_content"], icon["filename"])
            else:
                meta = generate_icon_metadata_from_image(
                    image_bytes=icon["raw_bytes"],
                    mime_type=icon.get("mime_type", "image/png"),
                    filename=icon["filename"],
                )

            enriched.append(
                {
                    **icon,
                    "name":        meta["name"],
                    "description": meta["description"],
                    "category":    meta["category"],
                    "style":       meta["style"],
                    "tags":        meta.get("tags", []),
                    "use_cases":   meta.get("use_cases", []),
                    "keywords":    meta.get("keywords", []),
                    "viewbox":     meta.get("viewbox"),
                    "llm_raw":     meta.get("_llm_raw", {}),
                }
            )
        except Exception as exc:
            err_msg = f"llm_enrich [{icon['filename']}]: {exc}"
            errors.append(err_msg)
            enriched.append({**icon, "error": err_msg})

    return {**state, "icons": enriched, "errors": errors}


# ── Node 3 : upload_r2 ────────────────────────────────────────────────────────
def upload_r2(state: SvgPipelineState) -> SvgPipelineState:
    """Upload each icon file bytes to Cloudflare R2 and record r2_key + public_url."""
    uploaded: list[SvgIconItem] = []
    errors: list[str] = list(state.get("errors", []))

    for icon in state["icons"]:
        if icon.get("error"):
            uploaded.append(icon)
            continue
        try:
            file_bytes = icon.get("raw_bytes")
            if not file_bytes and icon.get("svg_content"):
                file_bytes = icon["svg_content"].encode("utf-8")

            mime = icon.get("mime_type", "image/svg+xml")
            r2_key, public_url = upload_image_to_r2(
                file_bytes=file_bytes,
                original_filename=icon["filename"],
                content_type=mime,
                folder="icons",
            )
            uploaded.append({**icon, "r2_key": r2_key, "public_url": public_url})
        except Exception as exc:
            err_msg = f"upload_r2 [{icon['filename']}]: {exc}"
            errors.append(err_msg)
            uploaded.append({**icon, "error": err_msg})

    return {**state, "icons": uploaded, "errors": errors}


# ── Node 4 : save_db ──────────────────────────────────────────────────────────
def save_db(state: SvgPipelineState) -> SvgPipelineState:
    """
    Persist SvgIcon records to PostgreSQL.

    Runs synchronously (blocking) inside the LangGraph node — wraps the
    async SQLAlchemy session in asyncio.run() so it works in sync context.
    Uses upsert-by-slug logic: if the slug already exists, update it.
    """
    saved: list[SvgIconItem] = []
    errors: list[str] = list(state.get("errors", []))

    async def _upsert_icons(icons: list[SvgIconItem]) -> list[SvgIconItem]:
        results: list[SvgIconItem] = []
        async with async_session_factory() as session:
            async with session.begin():
                for icon in icons:
                    if icon.get("error"):
                        results.append(icon)
                        continue
                    try:
                        # Check for existing slug
                        stmt = select(SvgIcon).where(SvgIcon.slug == icon["slug"])
                        row = await session.execute(stmt)
                        existing: SvgIcon | None = row.scalar_one_or_none()

                        if existing:
                            # Update in-place
                            existing.name = icon["name"]
                            existing.description = icon["description"]
                            existing.category = IconCategory(icon["category"])
                            existing.style = IconStyle(icon["style"])
                            existing.tags = icon.get("tags", [])
                            existing.use_cases = icon.get("use_cases", [])
                            existing.keywords = icon.get("keywords", [])
                            existing.llm_raw = icon.get("llm_raw", {})
                            existing.file_format = icon.get("file_format", "svg")
                            existing.svg_content = icon.get("svg_content")
                            existing.r2_key = icon["r2_key"]
                            existing.public_url = icon["public_url"]
                            existing.file_size_bytes = icon.get("file_size_bytes")
                            existing.viewbox = icon.get("viewbox")
                            existing.updated_at = datetime.now(timezone.utc)
                            db_id = existing.id
                        else:
                            # Insert new record
                            record = SvgIcon(
                                id=str(uuid.uuid4()),
                                name=icon["name"],
                                slug=icon["slug"],
                                original_filename=icon["filename"],
                                file_format=icon.get("file_format", "svg"),
                                description=icon["description"],
                                category=IconCategory(icon["category"]),
                                style=IconStyle(icon["style"]),
                                tags=icon.get("tags", []),
                                use_cases=icon.get("use_cases", []),
                                keywords=icon.get("keywords", []),
                                llm_raw=icon.get("llm_raw", {}),
                                svg_content=icon.get("svg_content"),
                                r2_key=icon["r2_key"],
                                public_url=icon["public_url"],
                                file_size_bytes=icon.get("file_size_bytes"),
                                viewbox=icon.get("viewbox"),
                                vectorized=False,
                            )
                            session.add(record)
                            await session.flush()
                            await session.refresh(record)
                            db_id = record.id

                        results.append({**icon, "db_id": db_id})
                    except Exception as exc:
                        err_msg = f"save_db [{icon['filename']}]: {exc}"
                        errors.append(err_msg)
                        results.append({**icon, "error": err_msg})
        return results

    saved = asyncio.run(_upsert_icons([i for i in state["icons"]]))
    return {**state, "icons": saved, "errors": errors}


# ── Node 5 : embed_and_upsert ─────────────────────────────────────────────────
def embed_and_upsert(state: SvgPipelineState) -> SvgPipelineState:
    """
    Embed description + tags with Gemini and upsert vectors into Qdrant.
    Marks the DB record as vectorized after successful upsert.
    """
    errors: list[str] = list(state.get("errors", []))
    ready = [i for i in state["icons"] if not i.get("error") and i.get("db_id")]

    if not ready:
        return {**state, "processed_count": 0, "errors": errors}

    # Build rich text for embedding = description + tags + use_cases
    def _build_embed_text(icon: SvgIconItem) -> str:
        parts = [icon.get("description", "")]
        tags = icon.get("tags") or []
        use_cases = icon.get("use_cases") or []
        keywords = icon.get("keywords") or []
        if tags:
            parts.append("Tags: " + ", ".join(tags))
        if use_cases:
            parts.append("Use cases: " + "; ".join(use_cases))
        if keywords:
            parts.append("Keywords: " + ", ".join(keywords))
        return " | ".join(filter(None, parts))

    texts = [_build_embed_text(icon) for icon in ready]

    try:
        vectors = _embed_texts(texts)
    except Exception as exc:
        errors.append(f"embed_and_upsert: embedding failed: {exc}")
        return {**state, "processed_count": 0, "errors": errors}

    # Ensure Qdrant collection exists
    try:
        _ensure_svg_collection()
    except Exception as exc:
        errors.append(f"embed_and_upsert: Qdrant collection setup failed: {exc}")
        return {**state, "processed_count": 0, "errors": errors}

    client = get_qdrant_client()
    points: list[PointStruct] = []
    db_ids_to_mark: list[str] = []

    for icon, vector in zip(ready, vectors):
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, icon["db_id"]))
        payload = {
            "db_id":       icon["db_id"],
            "name":        icon.get("name"),
            "slug":        icon.get("slug"),
            "description": icon.get("description"),
            "category":    icon.get("category"),
            "style":       icon.get("style"),
            "tags":        icon.get("tags", []),
            "use_cases":   icon.get("use_cases", []),
            "keywords":    icon.get("keywords", []),
            "public_url":  icon.get("public_url"),
            "r2_key":      icon.get("r2_key"),
            "viewbox":     icon.get("viewbox"),
            "filename":    icon.get("filename"),
            "created_at":  datetime.now(timezone.utc).isoformat(),
        }
        points.append(PointStruct(id=point_id, vector=vector, payload=payload))
        db_ids_to_mark.append(icon["db_id"])

    try:
        client.upsert(collection_name=SVG_COLLECTION, points=points, wait=True)
    except Exception as exc:
        errors.append(f"embed_and_upsert: Qdrant upsert failed: {exc}")
        return {**state, "processed_count": 0, "errors": errors}

    # Mark records as vectorized in Postgres
    async def _mark_vectorized(db_ids: list[str]) -> None:
        async with async_session_factory() as session:
            async with session.begin():
                for db_id in db_ids:
                    stmt = select(SvgIcon).where(SvgIcon.id == db_id)
                    row = await session.execute(stmt)
                    rec: SvgIcon | None = row.scalar_one_or_none()
                    if rec:
                        rec.vectorized = True
                        rec.vectorized_at = datetime.now(timezone.utc)

    try:
        asyncio.run(_mark_vectorized(db_ids_to_mark))
    except Exception as exc:
        errors.append(f"embed_and_upsert: marking vectorized failed: {exc}")

    return {**state, "processed_count": len(points), "errors": errors}


# ── Node 6 : done ──────────────────────────────────────────────────────────────
def done(state: SvgPipelineState) -> SvgPipelineState:
    """Terminal node — compute final counts and pass state through."""
    total = len(state.get("icons", []))
    failed = sum(1 for i in state.get("icons", []) if i.get("error"))
    return {**state, "processed_count": total - failed, "failed_count": failed}


# ── Build the graph ────────────────────────────────────────────────────────────
def build_svg_ingest_graph() -> Any:
    graph = StateGraph(SvgPipelineState)

    graph.add_node("scan_files",        scan_files)
    graph.add_node("llm_enrich",        llm_enrich)
    graph.add_node("upload_r2",         upload_r2)
    graph.add_node("save_db",           save_db)
    graph.add_node("embed_and_upsert",  embed_and_upsert)
    graph.add_node("done",              done)

    graph.set_entry_point("scan_files")
    graph.add_edge("scan_files",       "llm_enrich")
    graph.add_edge("llm_enrich",       "upload_r2")
    graph.add_edge("upload_r2",        "save_db")
    graph.add_edge("save_db",          "embed_and_upsert")
    graph.add_edge("embed_and_upsert", "done")
    graph.add_edge("done",             END)

    return graph.compile()


# ── Singleton ──────────────────────────────────────────────────────────────────
svg_ingest_graph = build_svg_ingest_graph()
