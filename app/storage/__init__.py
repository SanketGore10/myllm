"""Storage package initialization."""

from app.storage.database import (
    Database,
    Session,
    Message,
    get_db,
    init_database,
    close_database,
)
from app.storage.cache import Cache, get_embedding_cache, get_template_cache

__all__ = [
    # Database
    "Database",
    "Session",
    "Message",
    "get_db",
    "init_database",
    "close_database",
    # Cache
    "Cache",
    "get_embedding_cache",
    "get_template_cache",
]
