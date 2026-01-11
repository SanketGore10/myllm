"""Core package initialization."""

from app.core.config import Settings, get_settings, reload_settings
from app.core.session import SessionManager, create_session_manager
from app.core.prompt import PromptBuilder, create_prompt_builder, detect_template_from_model_name

# Lazy import to avoid circular dependencies
# from app.core.runtime import RuntimeManager, get_runtime

__all__ = [
    # Config
    "Settings",
    "get_settings",
    "reload_settings",
    # Session
    "SessionManager",
    "create_session_manager",
    # Prompt
    "PromptBuilder",
    "create_prompt_builder",
    "detect_template_from_model_name",
    # Runtime (import directly from app.core.runtime when needed)
    # "RuntimeManager",
    # "get_runtime",
]
