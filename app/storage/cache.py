"""
In-memory caching for embeddings and templates.

Provides simple LRU-style caching with TTL expiration.
"""

import time
from typing import Any, Optional, Dict
from collections import OrderedDict
from threading import Lock

from app.utils.logging import get_logger

logger = get_logger(__name__)


class Cache:
    """Simple in-memory cache with TTL and LRU eviction."""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        """
        Initialize cache.
        
        Args:
            max_size: Maximum number of cached items
            default_ttl: Default TTL in seconds
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: OrderedDict[str, tuple[Any, float]] = OrderedDict()
        self._lock = Lock()
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
        
        Returns:
            Cached value or None if not found/expired
        """
        with self._lock:
            if key not in self._cache:
                logger.debug(f"Cache MISS: {key}")
                return None
            
            value, expiry = self._cache[key]
            
            # Check if expired
            if time.time() > expiry:
                logger.debug(f"Cache EXPIRED: {key}")
                del self._cache[key]
                return None
            
            # Move to end (LRU)
            self._cache.move_to_end(key)
            logger.debug(f"Cache HIT: {key}")
            return value
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (None = use default)
        """
        with self._lock:
            ttl = ttl if ttl is not None else self.default_ttl
            expiry = time.time() + ttl
            
            # Add/update entry
            self._cache[key] = (value, expiry)
            self._cache.move_to_end(key)
            
            # Evict oldest if over size limit
            while len(self._cache) > self.max_size:
                oldest_key = next(iter(self._cache))
                logger.debug(f"Cache EVICT (LRU): {oldest_key}")
                del self._cache[oldest_key]
            
            logger.debug(f"Cache SET: {key} (TTL={ttl}s)")
    
    def invalidate(self, key: str) -> None:
        """
        Remove key from cache.
        
        Args:
            key: Cache key to remove
        """
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                logger.debug(f"Cache INVALIDATE: {key}")
    
    def clear(self) -> None:
        """Clear all cached items."""
        with self._lock:
            self._cache.clear()
            logger.info("Cache cleared")
    
    def size(self) -> int:
        """Get current cache size."""
        return len(self._cache)
    
    def cleanup_expired(self) -> int:
        """
        Remove all expired entries.
        
        Returns:
            Number of removed entries
        """
        with self._lock:
            now = time.time()
            expired_keys = [
                key for key, (_, expiry) in self._cache.items()
                if now > expiry
            ]
            
            for key in expired_keys:
                del self._cache[key]
            
            if expired_keys:
                logger.info(f"Cache cleaned: removed {len(expired_keys)} expired entries")
            
            return len(expired_keys)


# Global cache instances
_embedding_cache: Optional[Cache] = None
_template_cache: Optional[Cache] = None


def get_embedding_cache() -> Cache:
    """Get global embedding cache."""
    global _embedding_cache
    if _embedding_cache is None:
        _embedding_cache = Cache(max_size=500, default_ttl=3600)  # 1 hour TTL
    return _embedding_cache


def get_template_cache() -> Cache:
    """Get global template cache."""
    global _template_cache
    if _template_cache is None:
        _template_cache = Cache(max_size=100, default_ttl=86400)  # 24 hour TTL
    return _template_cache
