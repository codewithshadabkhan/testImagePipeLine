"""
LangGraph vectorization pipeline.

Graph nodes
───────────
1. fetch_records   – pull all un-vectorized (or all) image records from Postgres
2. embed_records   – embed description text with Gemini text-embedding-004
3. upsert_qdrant   – upsert vectors + metadata payload into Qdrant
4. done            – summarize result

State
─────
PipelineState is a TypedDict that flows through every node.
"""

from __future__ import annotations

import uuid
from typing import Any, TypedDict

import google.genai as genai
from langgraph.graph import END, StateGraph
from qdrant_client.models import PointStruct

from app.core.config import get_settings
from app.services.qdrant_service import ensure_collection, get_qdrant_client

settings = get_settings()

# ── Embedding service with HuggingFace Fallback ────────────────────────────────
from app.services.embedding_service import embed_texts_with_fallback

# ── Pipeline state ─────────────────────────────────────────────────────────────
class PipelineState(TypedDict):
    records: list[dict[str, Any]]       # raw rows from Postgres
    embedded: list[dict[str, Any]]      # records + their vector
    upserted_count: int
    errors: list[str]


# ── Node 1 : fetch_records ─────────────────────────────────────────────────────
def fetch_records(state: PipelineState) -> PipelineState:
    """
    Receives records from outside (injected into initial state).
    This node validates/normalises them before embedding.
    """
    valid, errors = [], []
    for rec in state["records"]:
        if not rec.get("description", "").strip():
            errors.append(f"Skipped id={rec.get('id')} – empty description")
            continue
        valid.append(rec)

    return {**state, "records": valid, "errors": errors}

# ── Node 2 : embed_records ─────────────────────────────────────────────────────
def embed_records(state: PipelineState) -> PipelineState:
    """Batch-embed all descriptions with Gemini / Hugging Face fallback."""
    records = state["records"]
    if not records:
        return {**state, "embedded": [], "errors": state["errors"] + ["No records to embed."]}

    texts = [r["description"] for r in records]

    try:
        vectors, provider = embed_texts_with_fallback(texts)
    except Exception as exc:
        return {
            **state,
            "embedded": [],
            "errors": state["errors"] + [f"Embedding failed: {exc}"],
        }

    embedded = [
        {**rec, "_vector": vec}
        for rec, vec in zip(records, vectors)
    ]
    return {**state, "embedded": embedded}


# ── Node 3 : upsert_qdrant ─────────────────────────────────────────────────────
def upsert_qdrant(state: PipelineState) -> PipelineState:
    """Upsert embedded vectors with full metadata payload into Qdrant."""
    embedded = state["embedded"]
    if not embedded:
        return {**state, "upserted_count": 0}

    ensure_collection()
    client = get_qdrant_client()

    points = []
    for item in embedded:
        vector = item.pop("_vector")
        # Use record UUID as Qdrant point id (converted to int hash for qdrant)
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, item["id"]))
        payload = {
            "db_id":          item.get("id"),
            "title":          item.get("title"),
            "description":    item.get("description"),
            "category":       item.get("category"),
            "image_type":     item.get("image_type"),
            "public_url":     item.get("public_url"),
            "r2_key":         item.get("r2_key"),
            "file_size_bytes":item.get("file_size_bytes"),
            "width":          item.get("width"),
            "height":         item.get("height"),
            "mime_type":      item.get("mime_type"),
            "created_at":     str(item.get("created_at", "")),
        }
        points.append(PointStruct(id=point_id, vector=vector, payload=payload))

    try:
        client.upsert(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            points=points,
            wait=True,
        )
    except Exception as exc:
        return {
            **state,
            "upserted_count": 0,
            "errors": state["errors"] + [f"Qdrant upsert failed: {exc}"],
        }

    return {**state, "upserted_count": len(points)}


# ── Node 4 : done ──────────────────────────────────────────────────────────────
def done(state: PipelineState) -> PipelineState:
    """Terminal node — just passes state through."""
    return state


# ── Build the graph ────────────────────────────────────────────────────────────
def build_vectorize_graph() -> Any:
    graph = StateGraph(PipelineState)

    graph.add_node("fetch_records", fetch_records)
    graph.add_node("embed_records", embed_records)
    graph.add_node("upsert_qdrant", upsert_qdrant)
    graph.add_node("done", done)

    graph.set_entry_point("fetch_records")
    graph.add_edge("fetch_records", "embed_records")
    graph.add_edge("embed_records", "upsert_qdrant")
    graph.add_edge("upsert_qdrant", "done")
    graph.add_edge("done", END)

    return graph.compile()


# ── Singleton compiled graph ───────────────────────────────────────────────────
vectorize_graph = build_vectorize_graph()
