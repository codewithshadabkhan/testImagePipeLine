from fastapi import APIRouter
from app.api.v1.endpoints import images, pipeline

api_router = APIRouter()
api_router.include_router(images.router)
api_router.include_router(pipeline.router)
