"""
Rate Limiter distribuido (slowapi) con respaldo en Redis y clave por tenant/API key.
"""

from __future__ import annotations

import hashlib
import logging
from fastapi import Request
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.core import security
from app.core.config import settings

logger = logging.getLogger(__name__)


def tenant_or_ip_key(request: Request) -> str:
    """
    Identifica el origen del rate limit priorizando `tenant_id` o prefijo de
    API key / JWT sobre la IP remota para entornos multi-tenant tras proxy.
    """
    tenant_id = getattr(request.state, "tenant_id", None)
    if tenant_id:
        return f"tenant:{tenant_id}"

    api_key = request.headers.get("x-api-key")
    if api_key:
        digest = hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:16]
        return f"apikey:{digest}"

    auth_header = request.headers.get("authorization", "")
    if auth_header.lower().startswith("bearer "):
        token = auth_header.split(" ", 1)[1].strip()
        if token.startswith("px_"):
            digest = hashlib.sha256(token.encode("utf-8")).hexdigest()[:16]
            return f"apikey:{digest}"
        payload = security.decode_token(token)
        if payload and payload.get("sub"):
            return f"user:{payload['sub']}"

    return get_remote_address(request)


def _resolve_storage_uri() -> str:
    redis_url = getattr(settings, "REDIS_URL", None)
    if redis_url and settings.ENVIRONMENT != "test":
        return redis_url
    return "memory://"


limiter = Limiter(
    key_func=tenant_or_ip_key,
    storage_uri=_resolve_storage_uri(),
    in_memory_fallback_enabled=True,
)
