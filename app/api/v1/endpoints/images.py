"""
POST /api/v1/images/upload  – upload image + metadata
GET  /api/v1/images/        – fetch all images (with optional filters)
GET  /api/v1/images/{id}    – fetch single image by id
"""

from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.models.image_asset import ImageCategory, ImageType
from app.schemas.image import FetchResponse, ImageAssetRead, ImageUploadForm, UploadResponse
from app.services.image_service import (
    create_image_asset,
    get_all_image_assets,
    get_image_asset_by_id,
)
from app.services.r2_service import upload_image_to_r2
from app.utils.image_utils import (
    CONTENT_TYPE_TO_EXT,
    detect_image_type,
    get_image_dimensions,
    validate_image_file,
)


def _vectorize_single(record: dict) -> None:
    """Background task: vectorize one newly uploaded image record."""
    import asyncio
    from app.pipelines.vectorize import PipelineState, vectorize_graph

    state: PipelineState = {
        "records": [record],
        "embedded": [],
        "upserted_count": 0,
        "errors": [],
    }
    asyncio.run(vectorize_graph.ainvoke(state))


router = APIRouter(prefix="/images", tags=["Images"])


# ── 1. Upload endpoint ─────────────────────────────────────────────────────────
@router.post(
    "/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload an image with metadata",
    description=(
        "Accepts a multipart/form-data request with an image file and metadata fields. "
        "The image is stored in Cloudflare R2; metadata is persisted in PostgreSQL."
    ),
)
async def upload_image(
    # ── form fields ────────────────────────────────────────────
    title: str = Form(..., description="Short human-readable title"),
    description: str = Form(..., description="Required description of the image"),
    category: ImageCategory = Form(..., description="Image category"),
    # ── file ───────────────────────────────────────────────────
    file: UploadFile = File(..., description="The image file to upload (type is auto-detected)"),
    # ── dependencies ───────────────────────────────────────────
    background_tasks: BackgroundTasks = None,
    db: AsyncSession = Depends(get_db),
) -> UploadResponse:
    # 1. Read bytes
    raw_bytes = await file.read()
    content_type = file.content_type or "application/octet-stream"
    filename = file.filename or "upload"

    # 2. Validate file (size + allowed content-type)
    try:
        validate_image_file(content_type, len(raw_bytes))
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))

    # 3. Auto-detect image_type from content-type
    try:
        detected_image_type = detect_image_type(content_type)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))

    # 4. Extract dimensions
    dimensions = get_image_dimensions(raw_bytes)
    width, height = dimensions if dimensions else (None, None)

    # 5. Upload to Cloudflare R2
    try:
        r2_key, public_url = upload_image_to_r2(
            file_bytes=raw_bytes,
            original_filename=filename,
            content_type=content_type,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to upload image to R2: {exc}",
        )

    # 6. Persist metadata to PostgreSQL
    form = ImageUploadForm(
        title=title,
        description=description,
        category=category,
        image_type=detected_image_type,   # auto-detected
    )
    asset = await create_image_asset(
        db=db,
        form=form,
        r2_key=r2_key,
        public_url=public_url,
        file_size_bytes=len(raw_bytes),
        width=width,
        height=height,
        mime_type=content_type,
    )

    # 7. Queue auto-vectorization in the background
    if background_tasks is not None:
        record = {
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
        background_tasks.add_task(_vectorize_single, record)

    return UploadResponse(
        message="Image uploaded successfully. Vectorization queued.",
        data=ImageAssetRead.model_validate(asset),
    )


# ── 2a. Fetch all images ───────────────────────────────────────────────────────
@router.get(
    "/",
    response_model=FetchResponse,
    summary="Fetch all images",
    description=(
        "Returns a paginated list of image assets stored in PostgreSQL. "
        "Optionally filter by category or image_type."
    ),
)
async def fetch_images(
    category: Optional[ImageCategory] = None,
    image_type: Optional[ImageType] = None,
    skip: int = 0,
    limit: int = 50,
    db: AsyncSession = Depends(get_db),
) -> FetchResponse:
    total, items = await get_all_image_assets(
        db,
        category=category,
        image_type=image_type,
        skip=skip,
        limit=limit,
    )
    return FetchResponse(
        total=total,
        items=[ImageAssetRead.model_validate(item) for item in items],
    )


# ── 2b. Fetch single image ─────────────────────────────────────────────────────
@router.get(
    "/{image_id}",
    response_model=ImageAssetRead,
    summary="Fetch image by ID",
)
async def fetch_image_by_id(
    image_id: str,
    db: AsyncSession = Depends(get_db),
) -> ImageAssetRead:
    asset = await get_image_asset_by_id(db, image_id)
    if not asset:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Image not found.")
    return ImageAssetRead.model_validate(asset)
