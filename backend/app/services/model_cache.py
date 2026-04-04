import logging
import time
from threading import Lock
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class ModelCache:
    """
    Singleton cache para modelos ML con TTL y LRU eviction.
    Evita cargar el mismo modelo múltiples veces en memoria.
    """

    _instance: Optional["ModelCache"] = None
    _lock = Lock()

    def __init__(self, max_size: int = 10, default_ttl: int = 600):
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._max_size = max_size
        self._default_ttl = default_ttl

    @classmethod
    def get_instance(cls, max_size: int = 10, default_ttl: int = 600) -> "ModelCache":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(max_size, default_ttl)
            return cls._instance

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return None

            if time.time() > entry["expires_at"]:
                del self._cache[key]
                logger.debug("Model cache expired for key: %s", key)
                return None

            entry["last_access"] = time.time()
            entry["hits"] += 1
            logger.debug("Model cache hit for key: %s (hits: %d)", key, entry["hits"])
            return entry["model"]

    def set(self, key: str, model: Any, ttl: Optional[int] = None) -> None:
        with self._lock:
            if len(self._cache) >= self._max_size and key not in self._cache:
                self._evict_lru()

            ttl = ttl or self._default_ttl
            self._cache[key] = {
                "model": model,
                "created_at": time.time(),
                "last_access": time.time(),
                "hits": 0,
                "expires_at": time.time() + ttl,
            }
            logger.info("Model cached with key: %s (ttl: %ds)", key, ttl)

    def _evict_lru(self) -> None:
        if not self._cache:
            return

        lru_key = min(self._cache.keys(), key=lambda k: self._cache[k]["last_access"])
        del self._cache[lru_key]
        logger.info("Evicted LRU model from cache: %s", lru_key)

    def invalidate(self, key: str) -> None:
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                logger.info("Invalidated model cache: %s", key)

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            logger.info("Model cache cleared")

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "size": len(self._cache),
                "max_size": self._max_size,
                "keys": list(self._cache.keys()),
            }


def get_model_cache() -> ModelCache:
    return ModelCache.get_instance()
