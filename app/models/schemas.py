"""
Pydantic schemas for API requests/responses and internal data structures.

All data validation and serialization happens through these schemas.
"""

from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field, validator


# ============================================================================
# Message Schemas
# ============================================================================

class Message(BaseModel):
    """Chat message."""
    
    role: Literal["system", "user", "assistant"] = Field(..., description="Message role")
    content: str = Field(..., description="Message content")
    
    class Config:
        json_schema_extra = {
            "example": {
                "role": "user",
                "content": "Hello, how are you?"
            }
        }


# ============================================================================
# Inference Options
# ============================================================================

class InferenceOptions(BaseModel):
    """Inference generation options."""
    
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Nucleus sampling")
    top_k: Optional[int] = Field(default=None, ge=0, description="Top-K sampling")
    max_tokens: Optional[int] = Field(default=None, ge=1, description="Maximum tokens to generate")
    stop: Optional[List[str]] = Field(default=None, description="Stop sequences")
    repeat_penalty: Optional[float] = Field(default=None, ge=0.0, description="Repetition penalty")
    presence_penalty: Optional[float] = Field(default=None, ge=-2.0, le=2.0, description="Presence penalty")
    frequency_penalty: Optional[float] = Field(default=None, ge=-2.0, le=2.0, description="Frequency penalty")
    
    class Config:
        json_schema_extra = {
            "example": {
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 512
            }
        }


# ============================================================================
# Chat API Schemas
# ============================================================================

class ChatRequest(BaseModel):
    """Request schema for /api/chat endpoint."""
    
    model: str = Field(..., description="Model name to use")
    messages: List[Message] = Field(..., min_length=1, description="Conversation messages")
    session_id: Optional[str] = Field(default=None, description="Session ID to continue conversation")
    stream: bool = Field(default=True, description="Enable streaming response")
    options: Optional[InferenceOptions] = Field(default=None, description="Generation options")
    
    @validator("messages")
    def validate_messages(cls, messages):
        """Ensure at least one message and last message is from user."""
        if not messages:
            raise ValueError("At least one message required")
        # Last message should typically be from user, but we'll allow flexibility
        return messages
    
    class Config:
        json_schema_extra = {
            "example": {
                "model": "llama-3-8b",
                "messages": [
                    {"role": "user", "content": "Hello!"}
                ],
                "stream": True
            }
        }


class ChatResponse(BaseModel):
    """Response schema for /api/chat endpoint (non-streaming)."""
    
    message: Message = Field(..., description="Assistant's response")
    session_id: str = Field(..., description="Session ID for continuation")
    usage: Optional[Dict[str, int]] = Field(default=None, description="Token usage stats")
    
    class Config:
        json_schema_extra = {
            "example": {
                "message": {
                    "role": "assistant",
                    "content": "Hello! How can I help you?"
                },
                "session_id": "123e4567-e89b-12d3-a456-426614174000",
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 8,
                    "total_tokens": 18
                }
            }
        }


class ChatStreamChunk(BaseModel):
    """Streaming chunk for chat response."""
    
    token: Optional[str] = Field(default=None, description="Generated token")
    done: bool = Field(default=False, description="Whether generation is complete")
    session_id: Optional[str] = Field(default=None, description="Session ID (sent on completion)")
    full_text: Optional[str] = Field(default=None, description="Complete response (sent on completion)")
    error: Optional[str] = Field(default=None, description="Error message if failed")


# ============================================================================
# Generate API Schemas
# ============================================================================

class GenerateRequest(BaseModel):
    """Request schema for /api/generate endpoint."""
    
    model: str = Field(..., description="Model name to use")
    prompt: str = Field(..., min_length=1, description="Input prompt")
    stream: bool = Field(default=True, description="Enable streaming response")
    options: Optional[InferenceOptions] = Field(default=None, description="Generation options")
    
    class Config:
        json_schema_extra = {
            "example": {
                "model": "llama-3-8b",
                "prompt": "Once upon a time",
                "stream": False,
                "options": {
                    "max_tokens": 100,
                    "temperature": 0.8
                }
            }
        }


class GenerateResponse(BaseModel):
    """Response schema for /api/generate endpoint (non-streaming)."""
    
    text: str = Field(..., description="Generated text")
    usage: Optional[Dict[str, int]] = Field(default=None, description="Token usage stats")
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "Once upon a time, in a land far away...",
                "usage": {
                    "prompt_tokens": 4,
                    "completion_tokens": 10,
                    "total_tokens": 14
                }
            }
        }


class GenerateStreamChunk(BaseModel):
    """Streaming chunk for generate response."""
    
    token: Optional[str] = Field(default=None, description="Generated token")
    done: bool = Field(default=False, description="Whether generation is complete")
    full_text: Optional[str] = Field(default=None, description="Complete response (sent on completion)")
    error: Optional[str] = Field(default=None, description="Error message if failed")


# ============================================================================
# Model API Schemas
# ============================================================================

class ModelInfo(BaseModel):
    """Model information schema."""
    
    name: str = Field(..., description="Model name")
    family: str = Field(..., description="Model family (llama, mistral, etc.)")
    size_mb: Optional[int] = Field(default=None, description="Model file size in MB")
    quantization: str = Field(..., description="Quantization level")
    context_size: int = Field(..., description="Context window size")
    template: str = Field(default="chatml", description="Prompt template name")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Default parameters")
    loaded: bool = Field(default=False, description="Whether model is currently loaded")
    
    class Config:
        json_schema_extra = {
            "example": {
                "name": "llama-3-8b",
                "family": "llama",
                "size_mb": 4368,
                "quantization": "Q4_K_M",
                "context_size": 8192,
                "template": "llama3",
                "parameters": {
                    "temperature": 0.7,
                    "top_p": 0.9
                },
                "loaded": True
            }
        }


class ModelsListResponse(BaseModel):
    """Response schema for GET /api/models."""
    
    models: List[ModelInfo] = Field(..., description="List of available models")
    
    class Config:
        json_schema_extra = {
            "example": {
                "models": [
                    {
                        "name": "llama-3-8b",
                        "family": "llama",
                        "quantization": "Q4_K_M",
                        "context_size": 8192,
                        "template": "llama3",
                        "loaded": True
                    }
                ]
            }
        }


# ============================================================================
# Embeddings API Schemas
# ============================================================================

class EmbeddingRequest(BaseModel):
    """Request schema for /api/embeddings endpoint."""
    
    model: str = Field(..., description="Model name to use")
    input: str = Field(..., min_length=1, description="Text to embed")
    
    class Config:
        json_schema_extra = {
            "example": {
                "model": "llama-3-8b",
                "input": "Hello world"
            }
        }


class EmbeddingResponse(BaseModel):
    """Response schema for /api/embeddings endpoint."""
    
    embedding: List[float] = Field(..., description="Embedding vector")
    model: str = Field(..., description="Model used")
    
    class Config:
        json_schema_extra = {
            "example": {
                "embedding": [0.123, -0.456, 0.789],
                "model": "llama-3-8b"
            }
        }


# ============================================================================
# Internal Data Structures
# ============================================================================

class ModelConfig(BaseModel):
    """Model configuration (loaded from config.json)."""
    
    name: str
    family: str
    quantization: str
    context_size: int
    template: str = "chatml"
    parameters: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        json_schema_extra = {
            "example": {
                "name": "llama-3-8b",
                "family": "llama",
                "quantization": "Q4_K_M",
                "context_size": 8192,
                "template": "llama3",
                "parameters": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "repeat_penalty": 1.1
                }
            }
        }
