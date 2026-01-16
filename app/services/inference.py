"""
Inference service for executing model generation.
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
        self.model_loader = get_model_loader()
        self.settings = get_settings()
        self._last_model = None  # ✅ track last used model

    def infer(
        self,
        model_name: str,
        prompt: str,
        stop_tokens: Optional[List[str]] = None,
        options: Optional[InferenceOptions] = None,
        stream: bool = True,
    ) -> Iterator[str]:
        model = self.model_loader.get_or_load_model(model_name)
        self._last_model = model  # ✅ remember model instance

        final_options = self._merge_options(options)

        # Merge stop tokens (template + user)
        all_stop_tokens = list(stop_tokens or [])
        if final_options["stop"]:
            all_stop_tokens.extend(final_options["stop"])

        try:
            token_generator = model.generate(
                prompt=prompt,
                max_tokens=final_options["max_tokens"],
                temperature=final_options["temperature"],
                top_p=final_options["top_p"],
                top_k=final_options["top_k"],
                repeat_penalty=final_options["repeat_penalty"],
                stop=all_stop_tokens if all_stop_tokens else None,
                stream=stream,
            )

            for token in token_generator:
                yield token

        except Exception as e:
            logger.error(f"Inference failed for model {model_name}: {e}")
            raise InferenceError(f"Inference failed: {e}", model_name=model_name)

    def get_last_usage(self) -> Optional[Dict[str, int]]:
        """Return usage from last inference."""
        if self._last_model:
            return self._last_model.get_last_usage()
        return None

    def _merge_options(self, options: Optional[InferenceOptions]) -> Dict[str, Any]:
        merged = {
            "temperature": self.settings.default_temperature,
            "top_p": self.settings.default_top_p,
            "max_tokens": self.settings.default_max_tokens,
            "top_k": 40,
            "repeat_penalty": 1.1,
            "stop": None,
        }

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
# --------------------------------------
# Backward-compatibility factory
# --------------------------------------
_inference_service = None


def get_inference_service() -> InferenceService:
    """
    Get global inference service instance.
    Kept for backward compatibility (CLI, older imports).
    """
    global _inference_service
    if _inference_service is None:
        _inference_service = InferenceService()
    return _inference_service