## Image Pipeline API

A FastAPI + uv pipeline for uploading images to **Cloudflare R2**, storing metadata in **PostgreSQL**, auto-vectorizing descriptions with **Google Gemini (`gemini-embedding-001`)**, and performing semantic similarity search powered by **LangGraph** & **Qdrant**.

---

### Stack
| Layer | Technology |
|---|---|
| Runtime | Python 3.13 (managed by `uv`) |
| Framework | FastAPI + Uvicorn |
| ORM | SQLAlchemy 2 (async) |
| Migrations | Alembic |
| Storage | Cloudflare R2 via `boto3` (S3-compatible) |
| Database | PostgreSQL 16 + Adminer UI (Docker Compose) |
| Vision AI | Google Gemini 3.6 Flash (Auto-Image Description) |
| Orchestration | LangGraph |
| Embeddings | Google Gemini (`gemini-embedding-001` - 3072 dims) |
| Vector DB | Qdrant Cloud (with Payload Indexes) |

---

### Vectorization Pipeline Architecture

The image vectorization workflow uses **LangGraph** to process image records, generate dense embeddings, and store them along with rich metadata payloads in **Qdrant**.

#### 🔄 Pipeline Flow Diagram

```
                 POST /api/v1/images/upload
                             │
              Is description provided by user?
                   ├── NO ──► Gemini 3.6 Flash (Vision AI)
                   │          Generates brief image description
                   │          │
                   └── YES ───┴──► Cloudflare R2 (Image Upload)
                                           │
                                           ▼
                                 PostgreSQL (Saves Metadata + Description)
                                           │
                                           ▼
                             Background Task / Auto-Trigger
                                           │
                                           ▼
                             ┌─────────────────────────────────┐
                             │  LangGraph Vectorize Pipeline   │
                             └────────────────┬────────────────┘
                                              │
                                              ▼
                             ┌─────────────────────────────────┐
                             │  Node 1: fetch_records          │
                             │  Validates & prepares DB rows   │
                             └────────────────┬────────────────┘
                                              │
                                              ▼
                             ┌─────────────────────────────────┐
                             │  Node 2: embed_records          │
                             │  Gemini gemini-embedding-001    │
                             │  (Description -> 3072d vector)  │
                             └────────────────┬────────────────┘
                                              │
                                              ▼
                             ┌─────────────────────────────────┐
                             │  Node 3: upsert_qdrant          │
                             │  Upserts vector + payload       │
                             │  with auto-created indexes      │
                             └────────────────┬────────────────┘
                                              │
                                              ▼
                             ┌─────────────────────────────────┐
                             │  Node 4: done                   │
                             └─────────────────────────────────┘
```

#### 📌 Qdrant Payload & Indexing Strategy

To enable **O(log n)** hybrid filtering (semantic search + exact matching/range filtering), Qdrant automatically builds payload indexes on startup:

| Field | Qdrant Index Type | Purpose / Use Case |
|---|---|---|
| `category` | `KEYWORD` | Exact match filtering (nature, tech, etc.) |
| `image_type` | `KEYWORD` | Format filtering (png, jpeg, webp, etc.) |
| `mime_type` | `KEYWORD` | MIME type filtering |
| `public_url` | `KEYWORD` | Deduplication & direct point lookups |
| `file_size_bytes` | `INTEGER` | Min/Max file size range filtering |
| `width`, `height` | `INTEGER` | Image dimension filtering |
| `created_at` | `DATETIME` | Date range filtering |
| `title`, `description` | `TEXT` | Full-text BM25 search |

---

### Quickstart

#### 1. Configure Environment
Copy `.env.example` to `.env` and fill in your credentials:
```bash
cp .env.example .env
```

Ensure `.env` contains:
```ini
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5433/pipeline_db

R2_ACCOUNT_ID=your_account_id
R2_ACCESS_KEY_ID=your_access_key
R2_SECRET_ACCESS_KEY=your_secret_key
R2_BUCKET_NAME=your_bucket_name
R2_PUBLIC_URL=https://pub-xxxx.r2.dev

GEMINI_API_KEY=your_gemini_api_key

QDRANT_URL=https://your-cluster.qdrant.tech
QDRANT_API_KEY=your_qdrant_api_key
QDRANT_COLLECTION_NAME=image_assets
```

#### 2. Start PostgreSQL & Adminer
```bash
docker compose up -d
```
- **PostgreSQL**: `localhost:5433`
- **Adminer Web UI**: [http://localhost:8080](http://localhost:8080) (Server: `postgres`, DB: `pipeline_db`, User: `postgres`, Pass: `postgres`)

#### 3. Run Database Migrations
```bash
uv run alembic upgrade head
```

#### 4. Start the Dev Server
```bash
uv run uvicorn app.main:app --reload --port 8000
```
- API Docs (Swagger UI): [http://localhost:8000/docs](http://localhost:8000/docs)

---

### API Endpoints

#### 1. Upload Image & Auto-Vectorize
`POST /api/v1/images/upload`

Uploads the image file to R2, saves metadata to PostgreSQL, and queues a background task for LangGraph vectorization.

- **Form Fields (`multipart/form-data`)**:
  - `file`: Image binary (JPEG, PNG, WebP, GIF, AVIF)
  - `title` *(required)*: Short title
  - `description` *(required)*: Text description (used for vector embedding)
  - `category` *(required)*: `nature` | `technology` | `architecture` | `people` | `art` | `other`

*(Note: `image_type` is auto-detected from the file content-type)*

#### 2. Fetch Images (PostgreSQL)
`GET /api/v1/images/`

Paginated list of stored images with optional filtering (`category`, `image_type`, `skip`, `limit`).

#### 3. Bulk Vectorize Database Records
`POST /api/v1/pipeline/vectorize`

Triggers the full LangGraph pipeline to fetch all PostgreSQL records, embed any unindexed descriptions via Gemini, and upsert vectors + metadata to Qdrant.

#### 4. Semantic & Filtered Search (Qdrant)
`POST /api/v1/pipeline/search`

Executes a semantic similarity vector search combined with payload filters.

**Example Request Payload:**
```json
{
  "query": "scenic sunset over snow covered mountains",
  "top_k": 5,
  "score_threshold": 0.5,
  "category": "nature",
  "image_type": "png",
  "min_width": 1000,
  "min_height": 800
}
```

---

### Project Structure
```
app/
├── api/
│   └── v1/
│       ├── endpoints/
│       │   ├── images.py      ← Upload, Fetch endpoints + background vectorizer
│       │   └── pipeline.py    ├── Bulk vectorize & semantic search endpoints
│       └── router.py          ← Central API v1 router
├── core/
│   └── config.py              ← Pydantic Settings (.env configuration)
├── db/
│   ├── base.py                ← SQLAlchemy Declarative Base
│   └── session.py             ├── Async engine & session factory
├── models/
│   └── image_asset.py         ← PostgreSQL ORM model & enums
├── pipelines/
│   └── vectorize.py           ← LangGraph pipeline definition (nodes & graph compilation)
├── schemas/
│   └── image.py               ← Pydantic validation schemas
├── services/
│   ├── image_service.py       ← PostgreSQL Async CRUD operations
│   ├── qdrant_service.py      ├── Qdrant client singleton & payload indexing
│   ├── r2_service.py          ├── Cloudflare R2 boto3 client wrapper
│   └── search_service.py      └── Semantic vector search with payload filters
└── utils/
    └── image_utils.py         ← MIME validation & Pillow dimension extraction
```
