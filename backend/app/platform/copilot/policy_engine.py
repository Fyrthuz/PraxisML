"""
PolicyEngine determinista (servidor) para gobernar todas las acciones del Copilot y MCP.
Nunca delega decisiones de seguridad al prompt del LLM.
"""

from __future__ import annotations

from typing import Optional

from app.core.config import settings
from app.platform.copilot.contracts import (
    ActionRiskTier,
    CopilotTool,
    ExecutionIdentity,
    PolicyDecision,
    PolicyEvaluation,
)

ROLE_HIERARCHY = {
    "admin": 3,
    "editor": 2,
    "viewer": 1,
}


class PolicyEngine:
    """Evalúa RBAC, Tier de riesgo, entorno y presupuesto antes de ejecutar un tool."""

    def __init__(self, mutate_approval_threshold_usd: Optional[float] = None) -> None:
        self.mutate_approval_threshold_usd = (
            mutate_approval_threshold_usd
            if mutate_approval_threshold_usd is not None
            else settings.COPILOT_MUTATE_APPROVAL_THRESHOLD_USD
        )

    def evaluate(
        self,
        tool: CopilotTool,
        identity: ExecutionIdentity,
        estimated_cost_usd: float = 0.0,
        budget_remaining_usd: Optional[float] = None,
    ) -> PolicyEvaluation:
        # 1. Validación RBAC (Zero-Privilege Elevation)
        user_level = ROLE_HIERARCHY.get(identity.role, 0)
        required_level = ROLE_HIERARCHY.get(tool.min_role, 99)
        if user_level < required_level:
            return PolicyEvaluation(
                decision=PolicyDecision.DENY,
                reason=(
                    f"Rol insuficiente: '{tool.name}' requiere '{tool.min_role}', "
                    f"usuario tiene '{identity.role}'."
                ),
                risk_tier=tool.risk_tier,
                estimated_cost_usd=estimated_cost_usd,
            )

        # 2. Validación de presupuesto mensual del tenant/workspace
        if (
            budget_remaining_usd is not None
            and estimated_cost_usd > 0
            and estimated_cost_usd > budget_remaining_usd
        ):
            return PolicyEvaluation(
                decision=PolicyDecision.DENY,
                reason=(
                    f"Coste estimado (${estimated_cost_usd:.2f}) excede el presupuesto "
                    f"restante (${budget_remaining_usd:.2f})."
                ),
                risk_tier=tool.risk_tier,
                estimated_cost_usd=estimated_cost_usd,
            )

        # 3. Tier 3 (CRITICAL_PROD) siempre requiere aprobación humana explícita
        if tool.risk_tier >= ActionRiskTier.CRITICAL_PROD:
            return PolicyEvaluation(
                decision=PolicyDecision.REQUIRE_APPROVAL,
                reason=f"Acción Tier 3 ({tool.name}) requiere aprobación humana explícita.",
                risk_tier=tool.risk_tier,
                estimated_cost_usd=estimated_cost_usd,
            )

        # 4. Tier 2 (MUTATE_DEV): exige aprobación si afecta a producción o supera umbral FinOps
        if tool.risk_tier == ActionRiskTier.MUTATE_DEV:
            if identity.environment == "production" and tool.prod_requires_approval:
                return PolicyEvaluation(
                    decision=PolicyDecision.REQUIRE_APPROVAL,
                    reason=(
                        f"Mutación '{tool.name}' en entorno production requiere aprobación humana."
                    ),
                    risk_tier=tool.risk_tier,
                    estimated_cost_usd=estimated_cost_usd,
                )
            if estimated_cost_usd >= self.mutate_approval_threshold_usd:
                return PolicyEvaluation(
                    decision=PolicyDecision.REQUIRE_APPROVAL,
                    reason=(
                        f"Coste estimado (${estimated_cost_usd:.2f}) supera el umbral de "
                        f"auto-ejecución (${self.mutate_approval_threshold_usd:.2f})."
                    ),
                    risk_tier=tool.risk_tier,
                    estimated_cost_usd=estimated_cost_usd,
                )

        return PolicyEvaluation(
            decision=PolicyDecision.ALLOW,
            reason="Permitido por PolicyEngine.",
            risk_tier=tool.risk_tier,
            estimated_cost_usd=estimated_cost_usd,
        )
