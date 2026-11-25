"""
Router del Platform Copilot (/api/v1/copilot/*).

Expone introspección de tools y plugins activos filtrados por RBAC, junto con
los endpoints de chat e historial conversacional.
"""

from typing import Optional

from fastapi import APIRouter, Depends, Query, Request
from sqlalchemy.orm import Session

from app.api.deps import get_current_tenant, require_editor, require_viewer
from app.api.routes.v1.nlp.assistant import (
    ChatRequest,
    assistant_chat,
    clear_assistant_history,
    get_assistant_history,
)
from app.core.config import settings
from app.core.rate_limit import limiter
from app.database import get_db
from app.models.tenant import Tenant
from app.models.user import User
from app.platform.copilot.registry import get_copilot_tool_registry
from app.plugins.registry import get_plugin_registry

router = APIRouter()


@router.get("/tools")
def list_copilot_tools(
    namespace: Optional[str] = Query(default=None, description="Filtrar por namespace"),
    current_user: User = Depends(require_viewer),
):
    """Lista los tools registrados en CopilotToolRegistry permitidos para el rol del usuario."""
    registry = get_copilot_tool_registry()
    tools = registry.list_tools(role=current_user.role, namespace=namespace)
    return {
        "role": current_user.role,
        "count": len(tools),
        "tools": [t.to_metadata() for t in tools],
    }


@router.get("/plugins")
def list_active_plugins(
    _current_user: User = Depends(require_viewer),
):
    """Lista los plugins de dominio ML activos en la instancia."""
    registry = get_plugin_registry()
    plugins = registry.list_metadata()
    return {
        "count": len(plugins),
        "plugins": plugins,
    }


@router.post("/chat")
@limiter.limit(settings.RATE_LIMIT_LLM)
async def copilot_chat(
    request: Request,
    req: ChatRequest,
    user: User = Depends(require_editor),
    tenant: Tenant = Depends(get_current_tenant),
    db: Session = Depends(get_db),
):
    """Endpoint primario transversal de chat con el Platform Copilot."""
    return await assistant_chat(
        request=request, req=req, user=user, tenant=tenant, db=db
    )


@router.get("/history")
def copilot_history(
    limit: int = Query(20, ge=1, le=100),
    user: User = Depends(require_viewer),
    tenant: Tenant = Depends(get_current_tenant),
    db: Session = Depends(get_db),
):
    """Historial de conversaciones del Platform Copilot."""
    return get_assistant_history(limit=limit, _user=user, tenant=tenant, db=db)


@router.delete("/history")
def copilot_clear_history(
    user: User = Depends(require_editor),
    tenant: Tenant = Depends(get_current_tenant),
    db: Session = Depends(get_db),
):
    """Borra el historial de conversaciones del Platform Copilot."""
    return clear_assistant_history(_user=user, tenant=tenant, db=db)
