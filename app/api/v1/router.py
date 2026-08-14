from fastapi import APIRouter
from app.api.v1.endpoints import images, pipeline, svg_pipeline, pipeline_search

api_router = APIRouter()
api_router.include_router(images.router)
api_router.include_router(pipeline.router)
api_router.include_router(svg_pipeline.router)
api_router.include_router(pipeline_search.router)
