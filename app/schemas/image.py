from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, HttpUrl

from app.models.image_asset import ImageCategory, ImageType


# ── Upload request (form fields) ───────────────────────────────────────────────
class ImageUploadForm(BaseModel):
    """Mirrors the multipart/form-data fields sent with the image file."""

    title: str
    description: Optional[str] = None  # Auto-generated via Gemini Vision if omitted
    category: ImageCategory
    image_type: ImageType             # Populated by the endpoint via auto-detection

    model_config = ConfigDict(use_enum_values=True)


# ── Stored record (read) ───────────────────────────────────────────────────────
class ImageAssetRead(BaseModel):
    id: str
    title: str
    description: str
    category: ImageCategory
    image_type: ImageType
    r2_key: str
    public_url: str
    file_size_bytes: Optional[int]
    width: Optional[int]
    height: Optional[int]
    mime_type: Optional[str]
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True, use_enum_values=True)


# ── API response wrappers ──────────────────────────────────────────────────────
class UploadResponse(BaseModel):
    message: str
    data: ImageAssetRead


class FetchResponse(BaseModel):
    total: int
    items: list[ImageAssetRead]
