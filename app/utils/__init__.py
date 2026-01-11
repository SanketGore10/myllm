"""Utils package initialization."""

from app.utils.errors import (
    MyLLMError,
    ModelNotFoundError,
    ModelLoadError,
    InferenceError,
    SessionNotFoundError,
    ContextWindowExceededError,
    ConfigurationError,
    InvalidRequestError,
)
from app.utils.logging import setup_logging, get_logger, PerformanceLogger, set_request_id, clear_request_id
from app.utils.hardware import get_hardware_info, detect_gpu, HardwareInfo

__all__ = [
    # Errors
    "MyLLMError",
    "ModelNotFoundError",
    "ModelLoadError",
    "InferenceError",
    "SessionNotFoundError",
    "ContextWindowExceededError",
    "ConfigurationError",
    "InvalidRequestError",
    # Logging
    "setup_logging",
    "get_logger",
    "PerformanceLogger",
    "set_request_id",
    "clear_request_id",
    # Hardware
    "get_hardware_info",
    "detect_gpu",
    "HardwareInfo",
]
