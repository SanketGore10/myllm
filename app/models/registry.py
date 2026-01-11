"""
Model registry for discovering and managing available models.

Scans the models directory, loads configurations, and provides
model lookup and metadata access.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
from threading import Lock

from app.core.config import get_settings
from app.models.schemas import ModelInfo, ModelConfig
from app.utils.errors import ModelNotFoundError
from app.utils.logging import get_logger

logger = get_logger(__name__)


class ModelRegistry:
    """Registry for managing available models."""
    
    def __init__(self, models_dir: Path):
        """
        Initialize model registry.
        
        Args:
            models_dir: Path to models storage directory
        """
        self.models_dir = models_dir
        self._models: Dict[str, ModelInfo] = {}
        self._lock = Lock()
    
    def scan_models(self) -> List[ModelInfo]:
        """
        Scan models directory and load configurations.
        
        Returns:
            List of discovered models
        """
        with self._lock:
            self._models.clear()
            
            if not self.models_dir.exists():
                logger.warning(f"Models directory not found: {self.models_dir}")
                return []
            
            # Scan subdirectories
            for model_dir in self.models_dir.iterdir():
                if not model_dir.is_dir():
                    continue
                
                try:
                    model_info = self._load_model_info(model_dir)
                    if model_info:
                        self._models[model_info.name] = model_info
                        logger.info(f"Registered model: {model_info.name}")
                except Exception as e:
                    logger.error(f"Failed to load model from {model_dir}: {e}")
            
            logger.info(f"Scanned models directory: found {len(self._models)} models")
            return list(self._models.values())
    
    def _load_model_info(self, model_dir: Path) -> Optional[ModelInfo]:
        """
        Load model information from directory.
        
        Args:
            model_dir: Path to model directory
        
        Returns:
            ModelInfo or None if invalid
        """
        # Check for model file (*.gguf)
        gguf_files = list(model_dir.glob("*.gguf"))
        if not gguf_files:
            logger.warning(f"No GGUF file found in {model_dir}")
            return None
        
        model_file = gguf_files[0]
        
        # Load config.json
        config_file = model_dir / "config.json"
        if not config_file.exists():
            logger.warning(f"No config.json found in {model_dir}")
            # Create default config
            config = ModelConfig(
                name=model_dir.name,
                family="unknown",
                quantization="unknown",
                context_size=4096,
                template="chatml",
            )
        else:
            try:
                with open(config_file, "r") as f:
                    config_data = json.load(f)
                config = ModelConfig(**config_data)
            except Exception as e:
                logger.error(f"Failed to parse config.json in {model_dir}: {e}")
                return None
        
        # Get file size
        size_mb = model_file.stat().st_size // (1024 * 1024)
        
        model_info = ModelInfo(
            name=config.name,
            family=config.family,
            size_mb=size_mb,
            quantization=config.quantization,
            context_size=config.context_size,
            template=config.template,
            parameters=config.parameters,
            loaded=False,  # Will be updated by model loader
        )
        
        return model_info
    
    def get_model(self, name: str) -> ModelInfo:
        """
        Get model information by name.
        
        Args:
            name: Model name
        
        Returns:
            ModelInfo
        
        Raises:
            ModelNotFoundError: If model not found
        """
        with self._lock:
            if name not in self._models:
                raise ModelNotFoundError(name)
            return self._models[name]
    
    def get_model_path(self, name: str) -> Path:
        """
        Get path to model GGUF file.
        
        Args:
            name: Model name
        
        Returns:
            Path to model file
        
        Raises:
            ModelNotFoundError: If model not found
        """
        # Verify model exists
        self.get_model(name)
        
        model_dir = self.models_dir / name
        gguf_files = list(model_dir.glob("*.gguf"))
        
        if not gguf_files:
            raise ModelNotFoundError(f"Model file not found for {name}")
        
        return gguf_files[0]
    
    def get_model_config(self, name: str) -> ModelConfig:
        """
        Get model configuration.
        
        Args:
            name: Model name
        
        Returns:
            ModelConfig
        
        Raises:
            ModelNotFoundError: If model not found
        """
        model_dir = self.models_dir / name
        config_file = model_dir / "config.json"
        
        if not config_file.exists():
            # Return default config
            model_info = self.get_model(name)
            return ModelConfig(
                name=model_info.name,
                family=model_info.family,
                quantization=model_info.quantization,
                context_size=model_info.context_size,
                template=model_info.template,
                parameters=model_info.parameters,
            )
        
        with open(config_file, "r") as f:
            config_data = json.load(f)
        
        return ModelConfig(**config_data)
    
    def list_models(self) -> List[ModelInfo]:
        """
        Get list of all available models.
        
        Returns:
            List of ModelInfo
        """
        with self._lock:
            return list(self._models.values())
    
    def register_model(self, name: str, path: Path, config: ModelConfig) -> ModelInfo:
        """
        Manually register a model (used for dynamic model addition).
        
        Args:
            name: Model name
            path: Path to model file
            config: Model configuration
        
        Returns:
            ModelInfo
        """
        with self._lock:
            size_mb = path.stat().st_size // (1024 * 1024) if path.exists() else 0
            
            model_info = ModelInfo(
                name=config.name,
                family=config.family,
                size_mb=size_mb,
                quantization=config.quantization,
                context_size=config.context_size,
                template=config.template,
                parameters=config.parameters,
                loaded=False,
            )
            
            self._models[name] = model_info
            logger.info(f"Manually registered model: {name}")
            
            return model_info
    
    def update_model_loaded_status(self, name: str, loaded: bool) -> None:
        """
        Update model loaded status.
        
        Args:
            name: Model name
            loaded: Whether model is loaded
        """
        with self._lock:
            if name in self._models:
                self._models[name].loaded = loaded


# Global registry instance
_registry: Optional[ModelRegistry] = None


def get_registry() -> ModelRegistry:
    """
    Get global model registry instance.
    
    Returns:
        ModelRegistry
    """
    global _registry
    if _registry is None:
        settings = get_settings()
        _registry = ModelRegistry(settings.models_dir)
        _registry.scan_models()
    return _registry


def reload_registry() -> ModelRegistry:
    """
    Reload model registry (rescan models directory).
    
    Returns:
        ModelRegistry
    """
    global _registry
    settings = get_settings()
    _registry = ModelRegistry(settings.models_dir)
    _registry.scan_models()
    return _registry
