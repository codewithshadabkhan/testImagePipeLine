"""FastAPI application factory and startup/shutdown lifecycle."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.router import api_router
from app.core.config import get_settings
from app.db.base import Base
from app.db.session import engine
# Import models so Alembic / SQLAlchemy can discover them
import app.models.image_asset  # noqa: F401
import app.models.svg_icon     # noqa: F401

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create tables on startup (dev convenience — use Alembic in production)."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    await engine.dispose()


def create_app() -> FastAPI:
    app = FastAPI(
        title="Image Pipeline API",
        description=(
            "Upload images to Cloudflare R2 and store metadata in PostgreSQL. "
            "Built with FastAPI + SQLAlchemy + boto3."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS — tighten origins in production
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(api_router, prefix="/api/v1")

    @app.get("/health", tags=["Health"])
    async def health_check():
        return {"status": "ok", "env": settings.APP_ENV}

    return app


app = create_app()
