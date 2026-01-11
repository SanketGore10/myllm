"""
Models API endpoint for listing and managing models.

Handles GET /api/models for model discovery and information.
"""

from fastapi import APIRouter, HTTPException

from app.models.schemas import ModelsListResponse, ModelInfo
from app.models.registry import get_registry
from app.services.model_loader import get_model_loader
from app.utils.logging import get_logger
from app.utils.errors import ModelNotFoundError

logger = get_logger(__name__)

router = APIRouter(prefix="/api", tags=["models"])


@router.get("/models", response_model=ModelsListResponse)
async def list_models():
    """
    List all available models.
    
    Returns model metadata including whether each model is currently loaded.
    """
    logger.debug("List models request")
    
    try:
        registry = get_registry()
        model_loader = get_model_loader()
        
        # Get all models from registry
        models = registry.list_models()
        
        # Update loaded status
        loaded_models = model_loader.get_loaded_models()
        for model in models:
            model.loaded = model.name in loaded_models
        
        logger.info(f"Returning {len(models)} models ({len(loaded_models)} loaded)")
        
        return ModelsListResponse(models=models)
    
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(status_code=500, detail="Failed to list models")


@router.get("/models/{model_name}", response_model=ModelInfo)
async def get_model_info(model_name: str):
    """
    Get detailed information about a specific model.
    """
    logger.debug(f"Get model info: {model_name}")
    
    try:
        registry = get_registry()
        model_loader = get_model_loader()
        
        # Get model from registry
        model_info = registry.get_model(model_name)
        
        # Update loaded status
        model_info.loaded = model_loader.is_loaded(model_name)
        
        logger.info(f"Returning info for model: {model_name}")
        
        return model_info
    
    except ModelNotFoundError as e:
        logger.error(f"Model not found: {model_name}")
        raise HTTPException(status_code=404, detail=str(e))
    
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail="Failed to get model information")


@router.post("/models/{model_name}/load")
async def load_model(model_name: str):
    """
    Preload a model into memory.
    
    Useful for warming up the cache before first request.
    """
    logger.info(f"Preload model request: {model_name}")
    
    try:
        model_loader = get_model_loader()
        
        # Preload the model
        model_loader.preload_model(model_name)
        
        return {"status": "success", "message": f"Model '{model_name}' loaded"}
    
    except ModelNotFoundError as e:
        logger.error(f"Model not found: {model_name}")
        raise HTTPException(status_code=404, detail=str(e))
    
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")


@router.post("/models/{model_name}/unload")
async def unload_model(model_name: str):
    """
    Unload a model from memory.
    
    Frees up resources by removing model from cache.
    """
    logger.info(f"Unload model request: {model_name}")
    
    try:
        model_loader = get_model_loader()
        
        # Check if model is loaded
        if not model_loader.is_loaded(model_name):
            raise HTTPException(status_code=400, detail=f"Model '{model_name}' is not loaded")
        
        # Unload the model
        model_loader.unload_model(model_name)
        
        return {"status": "success", "message": f"Model '{model_name}' unloaded"}
    
    except HTTPException:
        raise
    
    except Exception as e:
        logger.error(f"Error unloading model: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to unload model: {e}")
