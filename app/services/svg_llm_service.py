"""
Gemini LLM service for analysing icon assets and generating rich metadata.

Supports two modes:
  1. SVG (text-based): parses path data + attributes, sends to Gemini text model
  2. Raster (PNG/JPG/WebP/ICO/GIF): sends image bytes to Gemini Vision model

Both modes return the same structured JSON payload:
  - name        : clean human-readable name
  - description : 2-3 sentence semantic description (used for embedding)
  - category    : one of the IconCategory enum values
  - style       : one of the IconStyle enum values
  - tags        : list of keyword strings
  - use_cases   : list of context / use-case strings
  - keywords    : list of synonyms / alternative search terms
"""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET

import google.genai as genai
from google.genai import types

from app.core.config import get_settings

settings = get_settings()

# Use a fast, capable text model for structured extraction
LLM_MODEL = "models/gemini-3.6-flash"

# ── Valid enum values (mirrors the ORM enums) ──────────────────────────────────
VALID_CATEGORIES = [
    "ui", "social", "communication", "media", "navigation", "finance",
    "nature", "technology", "travel", "health", "food", "weather",
    "security", "education", "business", "other",
]
VALID_STYLES = ["outline", "filled", "duotone", "flat", "other"]

# ── System prompt ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert SVG icon analyst and metadata generator.
You receive the filename and raw SVG markup of an icon, and you return a
structured JSON object with the following fields:

{
  "name": "Human-readable icon name (Title Case, 1-4 words)",
  "description": "2-3 sentence semantic description covering the icon's visual appearance, metaphor, and primary use-case. Write in plain English, no introductory meta-text.",
  "category": "<one of: ui, social, communication, media, navigation, finance, nature, technology, travel, health, food, weather, security, education, business, other>",
  "style": "<one of: outline, filled, duotone, flat, other>",
  "tags": ["keyword1", "keyword2", ...],
  "use_cases": ["specific use-case1", "specific use-case2", ...],
  "keywords": ["synonym1", "alternative term2", ...]
}

Rules:
- Return ONLY valid JSON — no markdown fences, no explanatory text.
- tags: 5-10 short lowercase keywords most relevant to the icon.
- use_cases: 3-6 specific contexts where this icon would be used in a UI.
- keywords: 5-8 synonyms or alternative search terms a developer might type.
- If the icon is ambiguous, infer from the filename and path geometry.
"""


# ── Helpers ────────────────────────────────────────────────────────────────────

def _extract_viewbox(svg_content: str) -> str | None:
    """Extract the viewBox attribute from SVG markup."""
    try:
        root = ET.fromstring(svg_content)
        return root.get("viewBox") or root.get("viewbox")
    except ET.ParseError:
        # Fallback: regex
        match = re.search(r'viewBox=["\']([^"\']+)["\']', svg_content, re.IGNORECASE)
        return match.group(1) if match else None


def _slug_to_name(slug: str) -> str:
    """Convert 'account-alert' → 'Account Alert' as a fallback name."""
    return " ".join(word.capitalize() for word in slug.replace("_", "-").split("-"))


def _parse_llm_json(raw_text: str) -> dict:
    """Strip markdown fences and parse JSON from the LLM response."""
    # Remove ```json ... ``` or ``` ... ``` wrappers if present
    text = re.sub(r"^```(?:json)?\s*", "", raw_text.strip(), flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


def _sanitize_payload(payload: dict, filename_stem: str) -> dict:
    """Validate and coerce LLM output to safe known values."""
    # Category
    cat = str(payload.get("category", "other")).lower()
    payload["category"] = cat if cat in VALID_CATEGORIES else "other"

    # Style
    sty = str(payload.get("style", "other")).lower()
    payload["style"] = sty if sty in VALID_STYLES else "other"

    # Ensure list fields
    for key in ("tags", "use_cases", "keywords"):
        val = payload.get(key)
        if not isinstance(val, list):
            payload[key] = []
        else:
            payload[key] = [str(v) for v in val if v]

    # Fallback name
    if not payload.get("name"):
        payload["name"] = _slug_to_name(filename_stem)

    # Fallback description
    if not payload.get("description"):
        payload["description"] = f"An SVG icon named {payload['name']}."

    return payload


# ── Main public function ───────────────────────────────────────────────────────

def generate_svg_metadata(svg_content: str, filename: str) -> dict:
    """
    Call Gemini LLM to generate structured metadata for a single SVG icon.

    Parameters
    ----------
    svg_content : str
        Raw SVG file content.
    filename : str
        Original filename (e.g. 'account-alert.svg') — used for context.

    Returns
    -------
    dict with keys: name, description, category, style, tags, use_cases,
                    keywords, plus the original viewbox.
    """
    filename_stem = filename.rsplit(".", 1)[0]

    # Build a concise user message — we only pass the first 4 KB of SVG
    # to avoid token overflows on complex icons
    svg_snippet = svg_content[:4096]

    user_message = (
        f"Filename: {filename}\n\n"
        f"SVG content:\n{svg_snippet}"
    )

    import time

    client = genai.Client(api_key=settings.GEMINI_API_KEY)

    # ── Retry loop for transient 503 / rate-limit errors ──────────────────────
    max_attempts = 3
    last_exc: Exception | None = None
    response = None
    for attempt in range(max_attempts):
        try:
            response = client.models.generate_content(
                model=LLM_MODEL,
                contents=[
                    types.Content(
                        role="user",
                        parts=[types.Part(text=user_message)],
                    )
                ],
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.2,
                    response_mime_type="application/json",
                ),
            )
            break  # success — exit retry loop
        except Exception as exc:
            last_exc = exc
            err_str = str(exc).lower()
            if any(code in err_str for code in ("503", "429", "unavailable", "rate")):
                wait_secs = 2 ** attempt  # 1s, 2s, 4s
                time.sleep(wait_secs)
            else:
                raise  # non-retriable — re-raise immediately
    if response is None:
        raise last_exc  # all retries exhausted

    raw_text = response.text.strip()

    try:
        payload = _parse_llm_json(raw_text)
    except (json.JSONDecodeError, ValueError):
        # Graceful fallback: minimal metadata from filename
        payload = {
            "name": _slug_to_name(filename_stem),
            "description": f"SVG icon: {_slug_to_name(filename_stem)}.",
            "category": "other",
            "style": "other",
            "tags": [],
            "use_cases": [],
            "keywords": [],
        }

    payload = _sanitize_payload(payload, filename_stem)

    # Attach viewbox for storage
    payload["viewbox"] = _extract_viewbox(svg_content)
    payload["_llm_raw"] = payload.copy()  # keep a raw copy before sanitization

    return payload


# ── Vision-based metadata (raster icons) ──────────────────────────────────────

RASTER_SYSTEM_PROMPT = """You are an expert icon analyst and metadata generator.
You receive an icon image and its filename, and you return structured JSON:

{
  "name": "Human-readable icon name (Title Case, 1-4 words)",
  "description": "2-3 sentence semantic description of the icon's visual appearance, metaphor, and primary use-case.",
  "category": "<one of: ui, social, communication, media, navigation, finance, nature, technology, travel, health, food, weather, security, education, business, other>",
  "style": "<one of: outline, filled, duotone, flat, other>",
  "tags": ["keyword1", "keyword2", ...],
  "use_cases": ["use-case1", "use-case2", ...],
  "keywords": ["synonym1", "alternative term2", ...]
}

Rules:
- Return ONLY valid JSON — no markdown fences, no extra text.
- tags: 5-10 short lowercase keywords.
- use_cases: 3-6 specific UI contexts where this icon would appear.
- keywords: 5-8 synonyms or alternative search terms.
"""


def generate_icon_metadata_from_image(
    image_bytes: bytes,
    mime_type: str,
    filename: str,
) -> dict:
    """
    Use Gemini Vision to analyse a raster icon (PNG/JPG/WebP/ICO/GIF)
    and generate the same structured metadata payload as generate_svg_metadata().

    Parameters
    ----------
    image_bytes : bytes
        Raw image file bytes.
    mime_type : str
        MIME type, e.g. 'image/png', 'image/jpeg', 'image/webp'.
    filename : str
        Original filename — used as a hint for the LLM.

    Returns
    -------
    dict with keys: name, description, category, style, tags, use_cases, keywords.
    """
    import time

    filename_stem = filename.rsplit(".", 1)[0]

    # Normalise ICO mime type — Gemini Vision doesn't support image/x-icon
    # Convert to PNG bytes on-the-fly using Pillow
    if mime_type in ("image/x-icon", "image/vnd.microsoft.icon", "image/ico"):
        try:
            from io import BytesIO
            from PIL import Image
            img = Image.open(BytesIO(image_bytes)).convert("RGBA")
            buf = BytesIO()
            img.save(buf, format="PNG")
            image_bytes = buf.getvalue()
            mime_type = "image/png"
        except Exception:
            pass  # fall through with original bytes

    client = genai.Client(api_key=settings.GEMINI_API_KEY)

    image_part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
    text_part  = types.Part(text=f"Filename: {filename}")

    max_attempts = 3
    last_exc: Exception | None = None
    response = None
    for attempt in range(max_attempts):
        try:
            response = client.models.generate_content(
                model=LLM_MODEL,
                contents=[types.Content(role="user", parts=[image_part, text_part])],
                config=types.GenerateContentConfig(
                    system_instruction=RASTER_SYSTEM_PROMPT,
                    temperature=0.2,
                    response_mime_type="application/json",
                ),
            )
            break
        except Exception as exc:
            last_exc = exc
            err_str = str(exc).lower()
            if any(c in err_str for c in ("503", "429", "unavailable", "rate")):
                time.sleep(2 ** attempt)
            else:
                raise
    if response is None:
        raise last_exc

    raw_text = response.text.strip()
    try:
        payload = _parse_llm_json(raw_text)
    except (json.JSONDecodeError, ValueError):
        payload = {
            "name": _slug_to_name(filename_stem),
            "description": f"Icon: {_slug_to_name(filename_stem)}.",
            "category": "other",
            "style": "other",
            "tags": [],
            "use_cases": [],
            "keywords": [],
        }

    payload = _sanitize_payload(payload, filename_stem)
    payload["viewbox"] = None  # raster icons have no viewBox
    payload["_llm_raw"] = payload.copy()
    return payload
