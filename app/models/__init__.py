"""Models package initialization."""

from app.models.schemas import (
    Message,
    InferenceOptions,
    ChatRequest,
    ChatResponse,
    ChatStreamChunk,
    GenerateRequest,
    GenerateResponse,
    GenerateStreamChunk,
    ModelInfo,
    ModelsListResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    ModelConfig,
)
from app.models.registry import ModelRegistry, get_registry, reload_registry

__all__ = [
    # Schemas
    "Message",
    "InferenceOptions",
    "ChatRequest",
    "ChatResponse",
    "ChatStreamChunk",
    "GenerateRequest",
    "GenerateResponse",
    "GenerateStreamChunk",
    "ModelInfo",
    "ModelsListResponse",
    "EmbeddingRequest",
    "EmbeddingResponse",
    "ModelConfig",
    # Registry
    "ModelRegistry",
    "get_registry",
    "reload_registry",
]
