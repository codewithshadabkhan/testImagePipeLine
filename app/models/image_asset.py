import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, Enum, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base

# ── Enums ──────────────────────────────────────────────────────────────────────
import enum


class ImageCategory(str, enum.Enum):
    nature = "nature"
    technology = "technology"
    architecture = "architecture"
    people = "people"
    art = "art"
    other = "other"


class ImageType(str, enum.Enum):
    jpeg = "jpeg"
    png = "png"
    webp = "webp"
    gif = "gif"
    avif = "avif"
    svg = "svg"


# ── ORM Model ──────────────────────────────────────────────────────────────────
class ImageAsset(Base):
    __tablename__ = "image_assets"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    category: Mapped[ImageCategory] = mapped_column(
        Enum(ImageCategory, name="imagecategory"), nullable=False
    )
    image_type: Mapped[ImageType] = mapped_column(
        Enum(ImageType, name="imagetype"), nullable=False
    )

    # R2 storage references
    r2_key: Mapped[str] = mapped_column(String(512), nullable=False, unique=True)
    public_url: Mapped[str] = mapped_column(String(1024), nullable=False)

    # File metadata
    file_size_bytes: Mapped[int | None] = mapped_column(nullable=True)
    width: Mapped[int | None] = mapped_column(nullable=True)
    height: Mapped[int | None] = mapped_column(nullable=True)
    mime_type: Mapped[str | None] = mapped_column(String(100), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    def __repr__(self) -> str:
        return f"<ImageAsset id={self.id!r} title={self.title!r}>"
