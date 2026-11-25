"""
CopilotToolRegistry — Registro dinámico de tools de plataforma y plugins.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from app.platform.copilot.contracts import ActionRiskTier, CopilotTool
from app.platform.copilot.policy_engine import ROLE_HIERARCHY

logger = logging.getLogger(__name__)


# ── Core Platform Tool Schemas & Handlers ─────────────────────────────────────


class _ListLimitArgs(BaseModel):
    limit: int = Field(default=20, ge=1, le=100)


class _PromoteModelArgs(BaseModel):
    model_id: str = Field(..., description="ID del modelo a promover")
    target_stage: str = Field(..., description="Staging | Production | Archived")


def _tool_list_models(args: _ListLimitArgs, **ctx: Any) -> Dict[str, Any]:
    from app.models.ml_model import MLModel

    db, identity = ctx["db"], ctx["identity"]
    rows = (
        db.query(MLModel)
        .filter(MLModel.tenant_id == identity.tenant_id, MLModel.is_active.is_(True))
        .limit(args.limit)
        .all()
    )
    return {
        "models": [
            {
                "id": m.id,
                "name": m.name,
                "version": m.version,
                "stage": m.stage,
                "mlflow_run_id": m.mlflow_run_id,
            }
            for m in rows
        ]
    }


def _tool_list_datasets(args: _ListLimitArgs, **ctx: Any) -> Dict[str, Any]:
    from app.models.dataset import Dataset

    db, identity = ctx["db"], ctx["identity"]
    rows = (
        db.query(Dataset)
        .filter(Dataset.tenant_id == identity.tenant_id)
        .limit(args.limit)
        .all()
    )
    return {
        "datasets": [
            {
                "id": d.id,
                "name": d.name,
                "file_type": d.file_type,
                "num_rows": d.num_rows,
                "num_columns": d.num_columns,
                "version": d.version,
            }
            for d in rows
        ]
    }


def _tool_get_costs(args: _ListLimitArgs, **ctx: Any) -> Dict[str, Any]:
    from app.observability.models import CostRecord

    db, identity = ctx["db"], ctx["identity"]
    rows = (
        db.query(CostRecord)
        .filter(CostRecord.tenant_id == identity.tenant_id)
        .order_by(CostRecord.period_start.desc())
        .limit(args.limit)
        .all()
    )
    return {
        "costs": [
            {
                "cost_type": r.cost_type,
                "amount": r.amount,
                "currency": r.currency,
                "model_id": r.model_id,
                "source": r.source,
            }
            for r in rows
        ]
    }


def _tool_list_serving_endpoints(args: _ListLimitArgs, **ctx: Any) -> Dict[str, Any]:
    from app.models.model_endpoint import ModelEndpoint

    db, identity = ctx["db"], ctx["identity"]
    rows = (
        db.query(ModelEndpoint)
        .filter(ModelEndpoint.tenant_id == identity.tenant_id)
        .limit(args.limit)
        .all()
    )
    return {
        "endpoints": [
            {
                "id": e.id,
                "name": e.name,
                "slug": e.slug,
                "model_id": e.model_id,
                "status": e.status,
                "traffic_split_pct": e.traffic_split_pct,
                "request_count": e.request_count,
                "avg_latency_ms": e.avg_latency_ms,
            }
            for e in rows
        ]
    }


def _tool_list_workspaces(args: _ListLimitArgs, **ctx: Any) -> Dict[str, Any]:
    from app.models.workspace import Workspace

    db, identity = ctx["db"], ctx["identity"]
    rows = (
        db.query(Workspace)
        .filter(Workspace.tenant_id == identity.tenant_id)
        .limit(args.limit)
        .all()
    )
    return {
        "workspaces": [
            {
                "id": w.id,
                "name": w.name,
                "slug": w.slug,
                "environment": w.environment,
                "default_compute_profile": w.default_compute_profile,
            }
            for w in rows
        ]
    }


def _tool_promote_model_placeholder(args: _PromoteModelArgs, **_ctx: Any) -> Dict[str, Any]:
    return {
        "status": "pending_approval",
        "model_id": args.model_id,
        "target_stage": args.target_stage,
    }


class CopilotToolRegistry:
    """Almacena todos los CopilotTool registrados por el Control Plane y los DomainPlugins."""

    def __init__(self) -> None:
        self._tools: Dict[str, CopilotTool] = {}

    def register(self, tool: CopilotTool) -> None:
        self._tools[tool.name] = tool
        logger.debug("Copilot tool registrado: %s (tier=%s)", tool.name, tool.risk_tier.name)

    def get(self, name: str) -> CopilotTool:
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' no encontrado en CopilotToolRegistry")
        return self._tools[name]

    def has(self, name: str) -> bool:
        return name in self._tools

    def clear(self) -> None:
        self._tools.clear()

    def list_tools(
        self,
        role: Optional[str] = None,
        namespace: Optional[str] = None,
    ) -> List[CopilotTool]:
        """Lista tools filtrando opcionalmente por rol RBAC máximo permitido y namespace."""
        tools = list(self._tools.values())
        if namespace:
            tools = [t for t in tools if t.namespace == namespace]
        if role:
            user_level = ROLE_HIERARCHY.get(role, 0)
            tools = [
                t for t in tools if ROLE_HIERARCHY.get(t.min_role, 99) <= user_level
            ]
        return sorted(tools, key=lambda t: (t.namespace, t.name))

    def list_openai_schemas(self, role: Optional[str] = None) -> List[Dict[str, Any]]:
        return [t.to_openai_tool_schema() for t in self.list_tools(role=role)]

    def register_core_platform_tools(self) -> None:
        """Registra los tools base del Control Plane (cross-domain)."""
        core_tools = [
            CopilotTool(
                name="models.list",
                namespace="models",
                description="Lista los modelos activos del tenant con su versión y stage.",
                args_schema=_ListLimitArgs,
                handler=_tool_list_models,
                min_role="viewer",
                risk_tier=ActionRiskTier.READ_METADATA,
                prod_requires_approval=False,
            ),
            CopilotTool(
                name="datasets.list",
                namespace="datasets",
                description="Lista los datasets registrados del tenant con metadatos de tamaño y esquema.",
                args_schema=_ListLimitArgs,
                handler=_tool_list_datasets,
                min_role="viewer",
                risk_tier=ActionRiskTier.READ_METADATA,
                prod_requires_approval=False,
            ),
            CopilotTool(
                name="observability.get_costs",
                namespace="observability",
                description="Consulta los registros de coste FinOps más recientes del tenant.",
                args_schema=_ListLimitArgs,
                handler=_tool_get_costs,
                min_role="viewer",
                risk_tier=ActionRiskTier.READ_METADATA,
                prod_requires_approval=False,
            ),
            CopilotTool(
                name="models.promote_stage",
                namespace="models",
                description="Propone promover un modelo a Production o archivarlo (requiere aprobación humana).",
                args_schema=_PromoteModelArgs,
                handler=_tool_promote_model_placeholder,
                min_role="editor",
                risk_tier=ActionRiskTier.CRITICAL_PROD,
                prod_requires_approval=True,
            ),
            CopilotTool(
                name="serving.list_endpoints",
                namespace="serving",
                description="Lista los endpoints dedicados de Model Serving del tenant con métricas de tráfico y latencia.",
                args_schema=_ListLimitArgs,
                handler=_tool_list_serving_endpoints,
                min_role="viewer",
                risk_tier=ActionRiskTier.READ_METADATA,
                prod_requires_approval=False,
            ),
            CopilotTool(
                name="workspaces.list",
                namespace="workspaces",
                description="Lista los workspaces configurados del tenant y sus perfiles de cómputo.",
                args_schema=_ListLimitArgs,
                handler=_tool_list_workspaces,
                min_role="viewer",
                risk_tier=ActionRiskTier.READ_METADATA,
                prod_requires_approval=False,
            ),
        ]
        for tool in core_tools:
            self.register(tool)


_tool_registry: Optional[CopilotToolRegistry] = None


def get_copilot_tool_registry() -> CopilotToolRegistry:
    """Devuelve la instancia singleton de CopilotToolRegistry."""
    global _tool_registry
    if _tool_registry is None:
        _tool_registry = CopilotToolRegistry()
    return _tool_registry
