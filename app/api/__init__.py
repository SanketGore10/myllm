"""API package initialization."""

from fastapi import APIRouter
from app.api import chat, generate, models, embeddings

# Create main API router
api_router = APIRouter()

# Include all endpoint routers
api_router.include_router(chat.router)
api_router.include_router(generate.router)
api_router.include_router(models.router)
api_router.include_router(embeddings.router)

__all__ = ["api_router"]
