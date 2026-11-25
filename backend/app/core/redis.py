import logging

from app.core.config import settings

logger = logging.getLogger(__name__)


def create_redis_client():
    """Cliente Redis asíncrono (para rutas async como WebSockets en streaming.py)."""
    try:
        import redis.asyncio as aioredis

        return aioredis.from_url(
            settings.CELERY_BROKER_URL,
            decode_responses=True,
            socket_connect_timeout=2,
            socket_timeout=2,
        )
    except Exception as e:
        logger.warning(
            "No se pudo conectar a Redis (async): %s. Usando almacenamiento en memoria.",
            e,
        )
        return None


def create_sync_redis_client():
    """Cliente Redis síncrono (para Celery tasks, CostCalculator y AlertEngine)."""
    try:
        import redis as sync_redis

        return sync_redis.from_url(
            settings.CELERY_BROKER_URL,
            decode_responses=True,
            socket_connect_timeout=2,
            socket_timeout=2,
        )
    except Exception as e:
        logger.warning(
            "No se pudo conectar a Redis (sync): %s. Usando almacenamiento en memoria.",
            e,
        )
        return None
