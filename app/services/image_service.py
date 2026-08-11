"""Image CRUD service — database operations."""

import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.image_asset import ImageAsset, ImageCategory, ImageType
from app.schemas.image import ImageUploadForm


async def create_image_asset(
    db: AsyncSession,
    form: ImageUploadForm,
    r2_key: str,
    public_url: str,
    file_size_bytes: Optional[int] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    mime_type: Optional[str] = None,
) -> ImageAsset:
    """Persist image metadata to PostgreSQL and return the saved record."""
    asset = ImageAsset(
        id=str(uuid.uuid4()),
        title=form.title,
        description=form.description,
        category=form.category,
        image_type=form.image_type,
        r2_key=r2_key,
        public_url=public_url,
        file_size_bytes=file_size_bytes,
        width=width,
        height=height,
        mime_type=mime_type,
    )
    db.add(asset)
    await db.flush()   # get the generated id without committing
    await db.refresh(asset)
    return asset


async def get_all_image_assets(
    db: AsyncSession,
    *,
    category: Optional[ImageCategory] = None,
    image_type: Optional[ImageType] = None,
    skip: int = 0,
    limit: int = 50,
) -> tuple[int, list[ImageAsset]]:
    """
    Fetch all image assets with optional filtering.

    Returns (total_count, items).
    """
    base_query = select(ImageAsset)
    count_query = select(func.count()).select_from(ImageAsset)

    if category:
        base_query = base_query.where(ImageAsset.category == category)
        count_query = count_query.where(ImageAsset.category == category)
    if image_type:
        base_query = base_query.where(ImageAsset.image_type == image_type)
        count_query = count_query.where(ImageAsset.image_type == image_type)

    total_result = await db.execute(count_query)
    total: int = total_result.scalar_one()

    items_result = await db.execute(
        base_query.order_by(ImageAsset.created_at.desc()).offset(skip).limit(limit)
    )
    items = list(items_result.scalars().all())

    return total, items


async def get_image_asset_by_id(
    db: AsyncSession, asset_id: str
) -> Optional[ImageAsset]:
    """Fetch a single image asset by primary key."""
    result = await db.execute(select(ImageAsset).where(ImageAsset.id == asset_id))
    return result.scalar_one_or_none()
