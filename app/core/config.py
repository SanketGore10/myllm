"""
Global configuration management.

Loads configuration from environment variables with validation.
Uses Pydantic Settings for type-safe configuration access.
"""

from pathlib import Path
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra='ignore',  # Ignore extra fields (e.g., LOG_LEVEL for legacy compat)
    )
    
    # Server Configuration
    host: str = Field(default="127.0.0.1", description="API server host")
    port: int = Field(default=8000, description="API server port")
    
    # Paths
    models_dir: Path = Field(default=Path("./models_data"), description="Models storage directory")
    db_path: Path = Field(default=Path("./myllm.db"), description="SQLite database path")
    
    # Default Inference Settings
    default_context_size: int = Field(default=4096, description="Default context window size")
    default_n_gpu_layers: int = Field(default=-1, description="Default GPU layers (-1 = all)")
    default_temperature: float = Field(default=0.7, description="Default temperature")
    default_top_p: float = Field(default=0.9, description="Default top_p")
    default_max_tokens: int = Field(default=512, description="Default max tokens")
    
    # Performance
    max_loaded_models: int = Field(default=3, description="Max models to keep in memory")
    enable_kv_cache: bool = Field(default=True, description="Enable KV cache")
    
    # Session Management
    session_retention_days: int = Field(default=30, description="Session retention period")
    max_session_messages: int = Field(default=1000, description="Max messages per session")
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Ensure directories exist
        self.models_dir.mkdir(parents=True, exist_ok=True)
        # Ensure db directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """
    Get global settings instance (singleton pattern).
    
    Returns:
        Settings instance
    """
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reload_settings() -> Settings:
    """
    Reload settings from environment (useful for testing).
    
    Returns:
        New Settings instance
    """
    global _settings
    _settings = Settings()
    return _settings
