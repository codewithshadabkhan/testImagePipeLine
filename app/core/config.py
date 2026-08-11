from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # App
    APP_ENV: str = "development"
    APP_SECRET_KEY: str = "change-me"

    # PostgreSQL (async)
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/pipeline_db"

    # Cloudflare R2  (S3-compatible)
    R2_ACCOUNT_ID: str
    R2_ACCESS_KEY_ID: str
    R2_SECRET_ACCESS_KEY: str
    R2_BUCKET_NAME: str
    R2_PUBLIC_URL: str  # base URL to construct public image links

    # Google Gemini (embeddings: gemini-embedding-001 → 3072 dims)
    GEMINI_API_KEY: str

    # Qdrant Cloud
    QDRANT_URL: str                              # e.g. https://xyz.qdrant.tech
    QDRANT_API_KEY: str
    QDRANT_COLLECTION_NAME: str = "image_assets"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
