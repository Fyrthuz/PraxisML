import logging

from app.core.config import settings

logger = logging.getLogger(__name__)


def create_redis_client():
    try:
        import redis.asyncio as aioredis
        return aioredis.from_url(
            settings.CELERY_BROKER_URL,
            decode_responses=True,
            socket_connect_timeout=2,
            socket_timeout=2,
        )
    except Exception as e:
        logger.warning("No se pudo conectar a Redis: %s. Usando almacenamiento en memoria.", e)
        return None
