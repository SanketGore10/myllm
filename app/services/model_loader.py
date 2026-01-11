"""
Model loader service with LRU caching.

Handles loading models on-demand, caching in memory,
and evicting least-recently-used models when cache is full.
"""

from pathlib import Path
from typing import Dict, Optional
from collections import OrderedDict
from threading import Lock

from app.models.registry import get_registry
from app.models.schemas import ModelInfo
from app.engine.llama_cpp import LlamaCppModel, load_model
from app.core.config import get_settings
from app.utils.errors import ModelNotFoundError, ModelLoadError
from app.utils.logging import get_logger

logger = get_logger(__name__)


class ModelLoader:
    """Model loader with LRU caching."""
    
    def __init__(self, max_models: int = 3):
        """
        Initialize model loader.
        
        Args:
            max_models: Maximum number of models to keep in memory
        """
        self.max_models = max_models
        self._loaded_models: OrderedDict[str, LlamaCppModel] = OrderedDict()
        self._lock = Lock()
        
        logger.info(f"ModelLoader initialized (max_models={max_models})")
    
    def get_or_load_model(self, model_name: str) -> LlamaCppModel:
        """
        Get model from cache or load it.
        
        Args:
            model_name: Name of the model
        
        Returns:
            Loaded LlamaCppModel instance
        
        Raises:
            ModelNotFoundError: If model doesn't exist
            ModelLoadError: If model fails to load
        """
        with self._lock:
            # Check if already loaded
            if model_name in self._loaded_models:
                logger.debug(f"Model cache HIT: {model_name}")
                # Move to end (mark as most recently used)
                self._loaded_models.move_to_end(model_name)
                return self._loaded_models[model_name]
            
            logger.info(f"Model cache MISS: {model_name}, loading...")
            
            # Get model info from registry
            registry = get_registry()
            model_info = registry.get_model(model_name)
            model_config = registry.get_model_config(model_name)
            model_path = registry.get_model_path(model_name)
            
            # Load the model
            try:
                model = load_model(
                    model_path=model_path,
                    n_ctx=model_config.context_size,
                    n_gpu_layers=None,  # Use default from settings
                )
                
                # Add to cache
                self._loaded_models[model_name] = model
                
                # Update registry status
                registry.update_model_loaded_status(model_name, True)
                
                # Evict oldest if over limit
                while len(self._loaded_models) > self.max_models:
                    oldest_name = next(iter(self._loaded_models))
                    logger.info(f"Evicting model (LRU): {oldest_name}")
                    self._unload_model(oldest_name)
                
                logger.info(
                    f"Model loaded successfully: {model_name} "
                    f"(cache size: {len(self._loaded_models)}/{self.max_models})"
                )
                
                return model
            
            except Exception as e:
                logger.error(f"Failed to load model {model_name}: {e}")
                raise ModelLoadError(model_name, str(e))
    
    def _unload_model(self, model_name: str) -> None:
        """
        Unload a model from cache (internal method, assumes lock held).
        
        Args:
            model_name: Name of model to unload
        """
        if model_name in self._loaded_models:
            model = self._loaded_models[model_name]
            model.close()
            del self._loaded_models[model_name]
            
            # Update registry status
            registry = get_registry()
            registry.update_model_loaded_status(model_name, False)
            
            logger.info(f"Model unloaded: {model_name}")
    
    def unload_model(self, model_name: str) -> None:
        """
        Manually unload a model.
        
        Args:
            model_name: Name of model to unload
        """
        with self._lock:
            self._unload_model(model_name)
    
    def preload_model(self, model_name: str) -> None:
        """
        Preload a model (warm up cache).
        
        Args:
            model_name: Name of model to preload
        """
        logger.info(f"Preloading model: {model_name}")
        self.get_or_load_model(model_name)
    
    def unload_all(self) -> None:
        """Unload all models."""
        with self._lock:
            model_names = list(self._loaded_models.keys())
            for name in model_names:
                self._unload_model(name)
            
            logger.info("All models unloaded")
    
    def get_loaded_models(self) -> list[str]:
        """
        Get list of currently loaded model names.
        
        Returns:
            List of model names
        """
        with self._lock:
            return list(self._loaded_models.keys())
    
    def is_loaded(self, model_name: str) -> bool:
        """
        Check if a model is currently loaded.
        
        Args:
            model_name: Model name
        
        Returns:
            True if loaded, False otherwise
        """
        with self._lock:
            return model_name in self._loaded_models


# Global model loader instance
_model_loader: Optional[ModelLoader] = None


def get_model_loader() -> ModelLoader:
    """
    Get global model loader instance.
    
    Returns:
        ModelLoader instance
    """
    global _model_loader
    if _model_loader is None:
        settings = get_settings()
        _model_loader = ModelLoader(max_models=settings.max_loaded_models)
    return _model_loader


def shutdown_model_loader() -> None:
    """Shutdown model loader (unload all models)."""
    global _model_loader
    if _model_loader:
        _model_loader.unload_all()
        _model_loader = None
