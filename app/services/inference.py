"""
Inference service for executing model generation.

Coordinates model loading and inference execution with parameter handling.
"""

from typing import Iterator, Optional, Dict, Any, List

from app.services.model_loader import get_model_loader
from app.models.schemas import InferenceOptions
from app.core.config import get_settings
from app.utils.errors import InferenceError
from app.utils.logging import get_logger

logger = get_logger(__name__)


class InferenceService:
    """Service for executing inference."""
    
    def __init__(self):
        """Initialize inference service."""
        self.model_loader = get_model_loader()
        self.settings = get_settings()
    
    def infer(
        self,
        model_name: str,
        prompt: str,
        stop_tokens: Optional[List[str]] = None,  # CRITICAL
        options: Optional[InferenceOptions] = None,
        stream: bool = True,
    ) -> Iterator[str]:
        """
        Execute inference on a prompt.
        
        Args:
            model_name: Name of model to use
            prompt: Input prompt
            stop_tokens: Stop sequences (CRITICAL for correct output)
            options: Generation options
            stream: Enable streaming
        
        Yields:
            Generated tokens
        
        Raises:
            InferenceError: If inference fails
        """
        # Get or load model
        model = self.model_loader.get_or_load_model(model_name)
        
        # Merge options with defaults
        final_options = self._merge_options(options)
        
        logger.debug(
            f"Starting inference: model={model_name}, "
            f"stream={stream}, temp={final_options['temperature']}, "
            f"stop_tokens={stop_tokens}"
        )
        
        # Execute inference
        try:
            token_generator = model.generate(
                prompt=prompt,
                max_tokens=final_options["max_tokens"],
                temperature=final_options["temperature"],
                top_p=final_options["top_p"],
                top_k=final_options["top_k"],
                repeat_penalty=final_options["repeat_penalty"],
                stop=stop_tokens,  # CRITICAL: Pass stop tokens
                stream=stream,
            )
            
            # Yield tokens from generator
            for token in token_generator:
                yield token
        
        except Exception as e:
            logger.error(f"Inference failed for model {model_name}: {e}")
            raise InferenceError(f"Inference failed: {e}", model_name=model_name)
    
    def _merge_options(self, options: Optional[InferenceOptions]) -> Dict[str, Any]:
        """
        Merge user options with system defaults.
        
        Args:
            options: User-provided options (may be None or partial)
        
        Returns:
            Complete options dictionary
        """
        # Start with defaults from settings
        merged = {
            "temperature": self.settings.default_temperature,
            "top_p": self.settings.default_top_p,
            "max_tokens": self.settings.default_max_tokens,
            "top_k": 40,
            "repeat_penalty": 1.1,
            "stop": None,
        }
        
        # Override with user options if provided
        if options:
            if options.temperature is not None:
                merged["temperature"] = options.temperature
            if options.top_p is not None:
                merged["top_p"] = options.top_p
            if options.max_tokens is not None:
                merged["max_tokens"] = options.max_tokens
            if options.top_k is not None:
                merged["top_k"] = options.top_k
            if options.repeat_penalty is not None:
                merged["repeat_penalty"] = options.repeat_penalty
            if options.stop is not None:
                merged["stop"] = options.stop
        
        return merged


def get_inference_service() -> InferenceService:
    """
    Get inference service instance.
    
    Returns:
        InferenceService instance
    """
    return InferenceService()
