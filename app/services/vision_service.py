"""Gemini Vision analysis service to generate image descriptions automatically."""

import google.genai as genai
from google.genai import types

from app.core.config import get_settings

settings = get_settings()

VISION_MODEL = "models/gemini-3.6-flash"


def generate_image_description(image_bytes: bytes, mime_type: str) -> str:
    """
    Analyzes an image using Gemini Vision and returns a concise description.
    """
    client = genai.Client(api_key=settings.GEMINI_API_KEY)

    image_part = types.Part.from_bytes(
        data=image_bytes,
        mime_type=mime_type,
    )

    prompt = (
        "Analyze this image and write a concise, clear description (1-3 sentences) "
        "highlighting key visual elements, objects, subject matter, colors, and context. "
        "Do not include introductory meta-text like 'This image shows' or 'The image is'."
    )

    response = client.models.generate_content(
        model=VISION_MODEL,
        contents=[image_part, prompt],
    )

    return response.text.strip()
