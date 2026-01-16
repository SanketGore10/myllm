"""
Logging configuration for MyLLM.

Provides structured logging with different formats for development and production.
Includes request ID tracking and performance metrics logging.
"""

import logging
import sys
import time
from contextvars import ContextVar
from typing import Optional
import json

# Context variable for request ID tracking
request_id_var: ContextVar[Optional[str]] = ContextVar("request_id", default=None)


class StructuredFormatter(logging.Formatter):
    """JSON formatter for production logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        
        # Add request ID if available
        request_id = request_id_var.get()
        if request_id:
            log_data["request_id"] = request_id
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # Add custom fields from extra
        if hasattr(record, "extra_fields"):
            log_data.update(record.extra_fields)
        
        return json.dumps(log_data)


class ColoredFormatter(logging.Formatter):
    """Colored formatter for development logging."""
    
    COLORS = {
        "DEBUG": "\033[36m",      # Cyan
        "INFO": "\033[32m",       # Green
        "WARNING": "\033[33m",    # Yellow
        "ERROR": "\033[31m",      # Red
        "CRITICAL": "\033[35m",   # Magenta
    }
    RESET = "\033[0m"
    
    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        
        # Add request ID if available
        request_id = request_id_var.get()
        if request_id:
            record.msg = f"[{request_id[:8]}] {record.msg}"
        
        return super().format(record)


def setup_logging(level: Optional[str] = None) -> None:
    """
    Configure application-wide logging.
    
    Default: WARNING (clean UX, Ollama-style)
    Debug mode: Set MYLLM_LOG_LEVEL=INFO or use --debug flag
    
    Args:
        level: Log level override (DEBUG, INFO, WARNING, ERROR, CRITICAL)
               If None, uses MYLLM_LOG_LEVEL env var or defaults to WARNING
    """
    import os
    
    # Determine log level (priority: argument > env var > default WARNING)
    if level is None:
        level = os.getenv("MYLLM_LOG_LEVEL", "WARNING").upper()
    
    # Validate level
    valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    if level not in valid_levels:
        level = "WARNING"
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,  # Override any existing config
    )
    
    # Silence noisy third-party libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured with level: {level}")


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for the given module name."""
    return logging.getLogger(name)


class PerformanceLogger:
    """Context manager for logging performance metrics."""
    
    def __init__(self, logger: logging.Logger, operation: str):
        self.logger = logger
        self.operation = operation
        self.start_time: Optional[float] = None
    
    def __enter__(self):
        self.start_time = time.time()
        self.logger.debug(f"Starting: {self.operation}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration_ms = (time.time() - self.start_time) * 1000
            if exc_type:
                self.logger.error(
                    f"Failed: {self.operation} ({duration_ms:.2f}ms)",
                    extra={"extra_fields": {"duration_ms": duration_ms, "success": False}}
                )
            else:
                self.logger.info(
                    f"Completed: {self.operation} ({duration_ms:.2f}ms)",
                    extra={"extra_fields": {"duration_ms": duration_ms, "success": True}}
                )


def set_request_id(request_id: str) -> None:
    """Set request ID for current context (used in API requests)."""
    request_id_var.set(request_id)


def clear_request_id() -> None:
    """Clear request ID from current context."""
    request_id_var.set(None)
