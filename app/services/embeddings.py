"""
Embeddings service for generating text embeddings.

Uses the model's embedding capabilities to generate vector representations.
"""

from typing import List
import hashlib

from app.services.model_loader import get_model_loader
from app.storage.cache import get_embedding_cache
from app.utils.errors import InferenceError
from app.utils.logging import get_logger

logger = get_logger(__name__)


class EmbeddingsService:
    """Service for generating embeddings."""
    
    def __init__(self):
        """Initialize embeddings service."""
        self.model_loader = get_model_loader()
        self.cache = get_embedding_cache()
    
    def generate_embedding(self, model_name: str, text: str) -> List[float]:
        """
        Generate embedding for text.
        
        Args:
            model_name: Name of model to use
            text: Input text
        
        Returns:
            Embedding vector as list of floats
        
        Raises:
            InferenceError: If embedding generation fails
        """
        # Check cache first
        cache_key = self._get_cache_key(model_name, text)
        cached_embedding = self.cache.get(cache_key)
        
        if cached_embedding:
            logger.debug(f"Embedding cache HIT for model {model_name}")
            return cached_embedding
        
        logger.debug(f"Embedding cache MISS for model {model_name}, generating...")
        
        # Get or load model
        model = self.model_loader.get_or_load_model(model_name)
        
        # Generate embedding
        try:
            embedding = model.embed(text)
            
            # Cache the result
            self.cache.set(cache_key, embedding, ttl=3600)
            
            logger.debug(
                f"Generated embedding: model={model_name}, "
                f"dim={len(embedding)}"
            )
            
            return embedding
        
        except Exception as e:
            logger.error(f"Embedding generation failed for model {model_name}: {e}")
            raise InferenceError(f"Embedding generation failed: {e}", model_name=model_name)
    
    def _get_cache_key(self, model_name: str, text: str) -> str:
        """
        Generate cache key for embedding.
        
        Args:
            model_name: Model name
            text: Input text
        
        Returns:
            Cache key string
        """
        # Hash text to create compact key
        text_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
        return f"embedding:{model_name}:{text_hash}"


def get_embeddings_service() -> EmbeddingsService:
    """
    Get embeddings service instance.
    
    Returns:
        EmbeddingsService instance
    """
    return EmbeddingsService()
