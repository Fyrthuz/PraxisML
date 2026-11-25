import logging
import time
from threading import Lock
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class BackgroundCache:
    """
    Cache para datos de background de SHAP por modelo.
    Evita recargar el mismo background data entre conexiones.
    """

    _instance: Optional["BackgroundCache"] = None
    _lock = Lock()

    def __init__(self, max_size: int = 20, default_ttl: int = 1800):
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._max_size = max_size
        self._default_ttl = default_ttl

    @classmethod
    def get_instance(
        cls, max_size: int = 20, default_ttl: int = 1800
    ) -> "BackgroundCache":
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
                logger.debug("Background cache expired for key: %s", key)
                return None

            entry["last_access"] = time.time()
            return entry["data"]

    def set(self, key: str, data: Any, ttl: Optional[int] = None) -> None:
        with self._lock:
            if len(self._cache) >= self._max_size and key not in self._cache:
                self._evict_oldest()

            ttl = ttl or self._default_ttl
            self._cache[key] = {
                "data": data,
                "created_at": time.time(),
                "last_access": time.time(),
                "expires_at": time.time() + ttl,
            }
            logger.info("Background data cached with key: %s (ttl: %ds)", key, ttl)

    def _evict_oldest(self) -> None:
        if not self._cache:
            return
        oldest_key = min(
            self._cache.keys(), key=lambda k: self._cache[k]["last_access"]
        )
        del self._cache[oldest_key]
        logger.info("Evicted oldest background from cache: %s", oldest_key)

    def invalidate(self, key: str) -> None:
        with self._lock:
            if key in self._cache:
                del self._cache[key]

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()


def get_background_cache() -> BackgroundCache:
    return BackgroundCache.get_instance()
