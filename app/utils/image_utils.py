"""Image-related utility helpers."""

from io import BytesIO
from typing import Optional

from PIL import Image

# Maximum allowed upload size: 20 MB
MAX_FILE_SIZE_BYTES = 20 * 1024 * 1024

ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/gif",
    "image/avif",
    "image/svg+xml",
}

CONTENT_TYPE_TO_EXT: dict[str, str] = {
    "image/jpeg": "jpeg",
    "image/png": "png",
    "image/webp": "webp",
    "image/gif": "gif",
    "image/avif": "avif",
    "image/svg+xml": "svg",
}

CONTENT_TYPE_TO_IMAGE_TYPE: dict[str, str] = {
    "image/jpeg": "jpeg",
    "image/png": "png",
    "image/webp": "webp",
    "image/gif": "gif",
    "image/avif": "avif",
    "image/svg+xml": "svg",
}


def detect_image_type(content_type: str) -> str:
    """
    Auto-detect ImageType enum value from MIME content-type.
    Returns the string value (e.g. 'png') or raises ValueError.
    """
    image_type = CONTENT_TYPE_TO_IMAGE_TYPE.get(content_type)
    if not image_type:
        raise ValueError(
            f"Cannot auto-detect image type from content-type '{content_type}'. "
            f"Supported types: {', '.join(sorted(ALLOWED_CONTENT_TYPES))}"
        )
    return image_type


def validate_image_file(content_type: str, size_bytes: int) -> None:
    """Raise ValueError if the upload does not pass basic validation."""
    if content_type not in ALLOWED_CONTENT_TYPES:
        raise ValueError(
            f"Unsupported content type '{content_type}'. "
            f"Allowed: {', '.join(sorted(ALLOWED_CONTENT_TYPES))}"
        )
    if size_bytes > MAX_FILE_SIZE_BYTES:
        raise ValueError(
            f"File too large ({size_bytes / 1024 / 1024:.1f} MB). "
            f"Maximum allowed size is {MAX_FILE_SIZE_BYTES // 1024 // 1024} MB."
        )


def get_image_dimensions(data: bytes) -> Optional[tuple[int, int]]:
    """Return (width, height) using Pillow or SVG parsing, or None on failure."""
    try:
        with Image.open(BytesIO(data)) as img:
            return img.size  # (width, height)
    except Exception:
        # Fallback for SVG parsing
        try:
            import re
            text = data.decode("utf-8", errors="ignore")
            # Try width and height attributes first
            w_match = re.search(r'width=["\'](\d+(?:\.\d+)?)', text)
            h_match = re.search(r'height=["\'](\d+(?:\.\d+)?)', text)
            if w_match and h_match:
                return (int(float(w_match.group(1))), int(float(h_match.group(1))))
            # Try viewBox
            vb_match = re.search(r'viewBox=["\']\s*0\s+0\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)', text, re.I)
            if vb_match:
                return (int(float(vb_match.group(1))), int(float(vb_match.group(2))))
        except Exception:
            pass
        return None
