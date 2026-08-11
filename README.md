## Image Pipeline API

A FastAPI + uv pipeline for uploading images to **Cloudflare R2** with metadata stored in **PostgreSQL**.

### Stack
| Layer | Technology |
|---|---|
| Runtime | Python 3.13 (managed by `uv`) |
| Framework | FastAPI + Uvicorn |
| ORM | SQLAlchemy 2 (async) |
| Migrations | Alembic |
| Storage | Cloudflare R2 via `boto3` (S3-compatible) |
| Database | PostgreSQL 16 (Docker Compose) |

---

### Quickstart

#### 1. Configure environment
```bash
cp .env.example .env
# fill in your R2 credentials and verify DATABASE_URL
```

#### 2. Start PostgreSQL
```bash
docker compose up -d
```

#### 3. Run migrations (first time / after schema changes)
```bash
uv run alembic revision --autogenerate -m "initial"
uv run alembic upgrade head
```

#### 4. Start the dev server
```bash
uv run uvicorn app.main:app --reload
```

The API is now available at **http://localhost:8000**  
Interactive docs: **http://localhost:8000/docs**

---

### API Endpoints

#### `POST /api/v1/images/upload`
Upload an image with metadata — multipart/form-data.

| Field | Type | Required | Notes |
|---|---|---|---|
| `file` | binary | ✅ | jpg / png / webp / gif / avif |
| `title` | string | ✅ | |
| `description` | string | | |
| `category` | enum | ✅ | nature, technology, architecture, people, art, other |
| `image_type` | enum | ✅ | jpeg, png, webp, gif, avif |

**Example (curl)**
```bash
curl -X POST http://localhost:8000/api/v1/images/upload \
  -F "file=@photo.jpg" \
  -F "title=My Photo" \
  -F "description=A nice photo" \
  -F "category=nature" \
  -F "image_type=jpeg"
```

#### `GET /api/v1/images/`
Fetch all images with optional query-string filters.

| Param | Type | Notes |
|---|---|---|
| `category` | enum | filter by category |
| `image_type` | enum | filter by type |
| `skip` | int | pagination offset (default 0) |
| `limit` | int | page size (default 50) |

#### `GET /api/v1/images/{image_id}`
Fetch a single image record by UUID.

---

### Project Structure
```
app/
  api/v1/endpoints/images.py   ← route handlers
  core/config.py               ← pydantic-settings
  db/
    base.py                    ← declarative base
    session.py                 ← async engine + session dep
  models/image_asset.py        ← ORM model
  schemas/image.py             ← Pydantic request/response schemas
  services/
    r2_service.py              ← Cloudflare R2 upload logic
    image_service.py           ← PostgreSQL CRUD
  utils/image_utils.py         ← validation + PIL helpers
  main.py                      ← FastAPI app factory
alembic/                       ← migration scripts
docker-compose.yml             ← PostgreSQL service
```
