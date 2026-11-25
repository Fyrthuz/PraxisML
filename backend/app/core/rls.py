"""
Helpers de PostgreSQL Row-Level Security (RLS) para aislamiento multi-tenant.
En dialectos distintos de PostgreSQL (por ejemplo SQLite en tests), opera como no-op.
"""

from __future__ import annotations

import logging
from sqlalchemy import text
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


def set_tenant_rls_context(db: Session, tenant_id: str) -> bool:
    """
    Configura `app.current_tenant_id` en el ámbito de la transacción actual
    de PostgreSQL (`SET LOCAL`) para reforzar el aislamiento por políticas RLS.
    """
    if not tenant_id:
        return False

    bind = db.get_bind()
    if bind is None or bind.dialect.name != "postgresql":
        return False

    try:
        db.execute(
            text("SELECT set_config('app.current_tenant_id', :tid, true)"),
            {"tid": str(tenant_id)},
        )
        return True
    except Exception as exc:
        logger.debug("No se pudo establecer contexto RLS para tenant %s: %s", tenant_id, exc)
        return False
