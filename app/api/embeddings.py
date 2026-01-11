"""
Embeddings API endpoint.

Handles POST /api/embeddings for generating text embeddings.
"""

from fastapi import APIRouter, HTTPException

from app.models.schemas import EmbeddingRequest, EmbeddingResponse
from app.core.runtime import get_runtime
from app.utils.logging import get_logger
from app.utils.errors import MyLLMError, ModelNotFoundError, InferenceError

logger = get_logger(__name__)

router = APIRouter(prefix="/api", tags=["embeddings"])


@router.post("/embeddings", response_model=EmbeddingResponse)
async def create_embedding(request: EmbeddingRequest):
    """
    Generate an embedding vector for input text.
    
    Embeddings can be used for semantic search, clustering, and similarity tasks.
    """
    logger.info(f"Embeddings request: model={request.model}, input_len={len(request.input)}")
    
    try:
        runtime = get_runtime()
        
        # Generate embedding
        embedding = runtime.embed(
            model_name=request.model,
            text=request.input,
        )
        
        logger.debug(f"Generated embedding: dim={len(embedding)}")
        
        return EmbeddingResponse(
            embedding=embedding,
            model=request.model,
        )
    
    except ModelNotFoundError as e:
        logger.error(f"Model not found: {e.model_name}")
        raise HTTPException(status_code=404, detail=str(e))
    
    except InferenceError as e:
        logger.error(f"Embedding error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    except MyLLMError as e:
        logger.error(f"MyLLM error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        logger.error(f"Unexpected error in embeddings endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
