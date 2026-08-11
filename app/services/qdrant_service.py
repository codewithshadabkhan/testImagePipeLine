"""Qdrant client singleton + collection bootstrap with payload indexes."""

from functools import lru_cache

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PayloadSchemaType,
    VectorParams,
)

from app.core.config import get_settings

settings = get_settings()

# gemini-embedding-001 outputs 3072-dimensional vectors
EMBEDDING_DIM = 3072


@lru_cache
def get_qdrant_client() -> QdrantClient:
    return QdrantClient(
        url=settings.QDRANT_URL,
        api_key=settings.QDRANT_API_KEY,
    )


# ── Payload index definitions ──────────────────────────────────────────────────
# Each entry: (field_name, PayloadSchemaType)
# keyword  → exact match / enum filters  (category, image_type, mime_type)
# integer  → range / equality filters    (file_size_bytes, width, height)
# datetime → date-range filters          (created_at)
# text     → full-text search index      (description, title)
PAYLOAD_INDEXES: list[tuple[str, PayloadSchemaType]] = [
    ("category",         PayloadSchemaType.KEYWORD),
    ("image_type",       PayloadSchemaType.KEYWORD),
    ("mime_type",        PayloadSchemaType.KEYWORD),
    ("public_url",       PayloadSchemaType.KEYWORD),  # dedup / point lookup
    ("file_size_bytes",  PayloadSchemaType.INTEGER),
    ("width",            PayloadSchemaType.INTEGER),
    ("height",           PayloadSchemaType.INTEGER),
    ("created_at",       PayloadSchemaType.DATETIME),
    ("title",            PayloadSchemaType.TEXT),      # full-text search
    ("description",      PayloadSchemaType.TEXT),      # full-text search
]


def create_payload_indexes(client: QdrantClient, collection_name: str) -> None:
    """
    Create payload indexes for all filterable / searchable fields.

    Why this matters
    ────────────────
    • keyword / integer / datetime indexes → enable O(log n) filtered vector
      search instead of O(n) full payload scan.  Critical once the collection
      grows beyond a few thousand points.

    • text indexes → enable Qdrant's built-in full-text BM25 search on
      title and description, which can be combined with vector search for
      hybrid retrieval.

    • Qdrant ignores duplicate create_payload_index calls, so this function
      is safe to call on every startup.
    """
    for field_name, schema_type in PAYLOAD_INDEXES:
        client.create_payload_index(
            collection_name=collection_name,
            field_name=field_name,
            field_schema=schema_type,
        )


def ensure_collection() -> None:
    """Create the Qdrant collection + all payload indexes if they don't exist."""
    client = get_qdrant_client()
    existing = {c.name for c in client.get_collections().collections}
    if settings.QDRANT_COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            vectors_config=VectorParams(
                size=EMBEDDING_DIM,
                distance=Distance.COSINE,
            ),
        )
    # Always ensure indexes exist (idempotent)
    create_payload_indexes(client, settings.QDRANT_COLLECTION_NAME)
