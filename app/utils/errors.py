"""
Custom exception classes for MyLLM.

This module defines the exception hierarchy used throughout the application.
All custom exceptions inherit from MyLLMError base class.
"""

from typing import Optional


class MyLLMError(Exception):
    """Base exception for all MyLLM errors."""
    
    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ModelNotFoundError(MyLLMError):
    """Raised when a requested model is not found in the registry."""
    
    def __init__(self, model_name: str):
        super().__init__(
            f"Model '{model_name}' not found in registry",
            {"model_name": model_name}
        )
        self.model_name = model_name


class ModelLoadError(MyLLMError):
    """Raised when a model fails to load."""
    
    def __init__(self, model_name: str, reason: str):
        super().__init__(
            f"Failed to load model '{model_name}': {reason}",
            {"model_name": model_name, "reason": reason}
        )
        self.model_name = model_name
        self.reason = reason


class InferenceError(MyLLMError):
    """Raised when inference execution fails."""
    
    def __init__(self, message: str, model_name: Optional[str] = None):
        super().__init__(
            message,
            {"model_name": model_name} if model_name else {}
        )
        self.model_name = model_name


class SessionNotFoundError(MyLLMError):
    """Raised when a session ID is not found in database."""
    
    def __init__(self, session_id: str):
        super().__init__(
            f"Session '{session_id}' not found",
            {"session_id": session_id}
        )
        self.session_id = session_id


class ContextWindowExceededError(MyLLMError):
    """Raised when prompt exceeds model's context window even after truncation."""
    
    def __init__(self, tokens: int, max_tokens: int):
        super().__init__(
            f"Prompt ({tokens} tokens) exceeds context window ({max_tokens} tokens)",
            {"tokens": tokens, "max_tokens": max_tokens}
        )
        self.tokens = tokens
        self.max_tokens = max_tokens


class ConfigurationError(MyLLMError):
    """Raised when configuration is invalid or missing."""
    
    def __init__(self, message: str):
        super().__init__(f"Configuration error: {message}")


class InvalidRequestError(MyLLMError):
    """Raised when API request validation fails."""
    
    def __init__(self, message: str, field: Optional[str] = None):
        details = {"field": field} if field else {}
        super().__init__(f"Invalid request: {message}", details)
        self.field = field
