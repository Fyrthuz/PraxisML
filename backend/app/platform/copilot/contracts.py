"""
Contratos del Platform Copilot: niveles de riesgo (4-Tier), identidad de ejecución y tools tipados.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Any, Callable, Dict, Optional, Type

from pydantic import BaseModel


class ActionRiskTier(IntEnum):
    """Niveles de riesgo para gobernanza de acciones del agente."""

    READ_METADATA = 0      # Lectura de metadatos/estado -> auto-ejecutable
    SAFE_COMPUTE = 1       # SQL read-only sandbox, búsqueda semántica, estimaciones -> auto-ejecutable
    MUTATE_DEV = 2         # Mutación/cómputo en dev/staging -> pre-flight budget check
    CRITICAL_PROD = 3      # Producción, borrados, permisos, cuotas -> aprobación humana obligatoria


class PolicyDecision(str, Enum):
    ALLOW = "allow"
    REQUIRE_APPROVAL = "require_approval"
    DENY = "deny"


@dataclass(frozen=True)
class ExecutionIdentity:
    """Identidad propagada desde el usuario autenticado (Zero-Privilege Elevation)."""

    user_id: str
    tenant_id: str
    role: str
    workspace_id: Optional[str] = None
    environment: str = "development"


@dataclass
class PolicyEvaluation:
    """Resultado determinista emitido por PolicyEngine antes de ejecutar un tool."""

    decision: PolicyDecision
    reason: str
    risk_tier: ActionRiskTier
    estimated_cost_usd: float = 0.0


@dataclass
class CopilotTool:
    """Definición tipada de una acción ejecutable por el Platform Copilot o MCP."""

    name: str
    namespace: str
    description: str
    args_schema: Type[BaseModel]
    handler: Callable[..., Any]
    min_role: str = "viewer"
    risk_tier: ActionRiskTier = ActionRiskTier.READ_METADATA
    prod_requires_approval: bool = True
    cost_estimator: Optional[Callable[[BaseModel], float]] = None

    def to_openai_tool_schema(self) -> Dict[str, Any]:
        """Exporta el tool al formato estándar OpenAI/vLLM function calling."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.args_schema.model_json_schema(),
            },
        }

    def to_metadata(self) -> Dict[str, Any]:
        """Metadatos públicos para introspección (/api/v1/copilot/tools)."""
        return {
            "name": self.name,
            "namespace": self.namespace,
            "description": self.description,
            "min_role": self.min_role,
            "risk_tier": int(self.risk_tier),
            "risk_tier_name": self.risk_tier.name,
            "prod_requires_approval": self.prod_requires_approval,
            "parameters": self.args_schema.model_json_schema(),
        }
