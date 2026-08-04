"""
Schemas Pydantic para los endpoints de observabilidad:
usage, costs, drift, alertas y health.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


# ── Usage ────────────────────────────────────────────────────────────────

class UsageQueryParams(BaseModel):
    tenant_id: Optional[str] = None
    domain: Optional[str] = None
    metric: Optional[str] = None
    from_: Optional[datetime] = Field(None, alias="from")
    to: Optional[datetime] = None
    group_by: Optional[str] = None
    limit: int = 100


class UsagePointResponse(BaseModel):
    recorded_at: datetime
    metric_name: str
    metric_value: float
    tags: Optional[dict] = None

    model_config = {"from_attributes": True}


class UsageAggregation(BaseModel):
    metric: str
    avg: Optional[float] = None
    p50: Optional[float] = None
    p95: Optional[float] = None
    p99: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None
    count: int = 0


class UsageResponse(BaseModel):
    tenant_id: str
    model_id: Optional[str] = None
    metrics: list[UsagePointResponse]
    aggregation: Optional[UsageAggregation] = None


# ── Costs ────────────────────────────────────────────────────────────────

class CostQueryParams(BaseModel):
    tenant_id: Optional[str] = None
    period: str = "monthly"
    cost_type: Optional[str] = None
    from_: Optional[datetime] = Field(None, alias="from")
    to: Optional[datetime] = None


class ModelCostBreakdown(BaseModel):
    model_id: str
    total_amount: float
    percentage: float

    model_config = {"from_attributes": True}


class CostBreakdownResponse(BaseModel):
    tenant_id: str
    cost_type: str
    total_amount: float
    percentage: float
    model_breakdown: Optional[list[ModelCostBreakdown]] = None


class CostForecastResponse(BaseModel):
    projected_month_end_usd: Optional[float] = None
    budget_usd: Optional[float] = None
    overrun_risk_pct: Optional[float] = None
    daily_trend_usd: Optional[float] = None


# ── Drift ────────────────────────────────────────────────────────────────

class DriftPointResponse(BaseModel):
    recorded_at: datetime
    drift_type: str
    psi: Optional[float] = None
    ks: Optional[float] = None
    is_anomaly: bool = False

    model_config = {"from_attributes": True}


class DriftHistoryResponse(BaseModel):
    model_id: str
    drift_points: list[DriftPointResponse]


# ── Alerts ───────────────────────────────────────────────────────────────

class AlertCreate(BaseModel):
    alert_type: str = Field(
        ..., description="cost_spike | latency_degradation | drift_critical | ..."
    )
    metric_name: str
    threshold: float
    comparison: str = Field(..., description="gt | lt | gte | lte | pct_change")
    cooldown_min: int = 60
    channels: list[dict] = Field(
        ..., description='[{type: "webhook", url: "..."}, {type: "email", to: "..."}]'
    )


class AlertUpdate(BaseModel):
    threshold: Optional[float] = None
    comparison: Optional[str] = None
    cooldown_min: Optional[int] = None
    channels: Optional[list[dict]] = None
    enabled: Optional[bool] = None


class AlertResponse(BaseModel):
    id: str
    tenant_id: str
    alert_type: str
    metric_name: str
    threshold: float
    comparison: str
    cooldown_min: int
    channels: list[dict]
    enabled: bool
    created_at: datetime

    model_config = {"from_attributes": True}


# ── Health ───────────────────────────────────────────────────────────────

class HealthComponent(BaseModel):
    name: str
    score: float
    status: str  # healthy | warning | critical


class HealthScoreResponse(BaseModel):
    model_id: str
    health_score: float = Field(..., ge=0, le=100)
    components: list[HealthComponent]


# ── Report ───────────────────────────────────────────────────────────────

class MonthlyReportResponse(BaseModel):
    tenant_id: str
    month: str
    format: str
    content_type: str
    download_url: Optional[str] = None
