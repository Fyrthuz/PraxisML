"""
Router de observabilidad: usage, costs, pricing, drift history, alerts, health.

Endpoints:
  GET    /usage                     — Métricas agregadas del tenant
  GET    /usage/models/{model_id}   — Métricas de un modelo
  GET    /costs                     — Desglose de costes
  GET    /costs/forecast            — Predicción de coste mensual
  POST   /pricing                  — Crear/actualizar precio (admin)
  GET    /pricing                  — Listar precios activos
  GET    /pricing/history          — Historial de cambios de precio
  GET    /drift/history/{model_id} — Serie temporal de drift
  POST   /alerts                   — Crear alerta
  GET    /alerts                   — Listar alertas del tenant
  PATCH  /alerts/{alert_id}        — Actualizar alerta
  DELETE /alerts/{alert_id}        — Eliminar alerta
  GET    /models/{model_id}/health — Health Score compuesto
"""

import logging
from datetime import datetime, timezone, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.api.deps import (
    get_db,
    get_current_tenant,
    get_cost_calculator_dep,
    require_admin,
    require_editor,
    require_viewer,
)
from app.models.tenant import Tenant
from app.models.user import User
from app.observability.models import (
    UsageMetric,
    CostRecord,
    DriftHistory,
    AlertConfig,
    ResourcePricing,
)
from app.observability.cost_calculator import CostCalculator
from app.schemas.observability import (
    UsageResponse,
    UsagePointResponse,
    UsageAggregation,
    CostForecastResponse,
    DriftHistoryResponse,
    DriftPointResponse,
    AlertCreate,
    AlertUpdate,
    AlertResponse,
    HealthScoreResponse,
    HealthComponent,
)
from app.schemas.pricing import (
    ResourcePricingCreate,
    ResourcePricingResponse,
    ResourcePricingHistoryResponse,
)

router = APIRouter(tags=["Observability"])
logger = logging.getLogger(__name__)


# ── Usage ─────────────────────────────────────────────────────────────────


@router.get("/usage", response_model=UsageResponse)
def get_usage(
    domain: Optional[str] = Query(None),
    metric: Optional[str] = Query(None),
    limit: int = Query(100, le=1000),
    tenant: Tenant = Depends(get_current_tenant),
    db: Session = Depends(get_db),
    _user: User = Depends(require_editor),
):
    """Métricas agregadas del tenant."""
    query = db.query(UsageMetric).filter(UsageMetric.tenant_id == tenant.id)
    if domain:
        query = query.filter(UsageMetric.domain == domain)
    if metric:
        query = query.filter(UsageMetric.metric_name == metric)
    items = query.order_by(UsageMetric.recorded_at.desc()).limit(limit).all()

    points = [
        UsagePointResponse(
            recorded_at=m.recorded_at,
            metric_name=m.metric_name,
            metric_value=m.metric_value,
            tags=m.tags,
        )
        for m in items
    ]

    agg = None
    if metric and items:
        values = [m.metric_value for m in items]
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        agg = UsageAggregation(
            metric=metric,
            avg=sum(values) / n,
            p50=sorted_vals[n // 2],
            p95=sorted_vals[int(n * 0.95)],
            p99=sorted_vals[int(n * 0.99)],
            min=min(values),
            max=max(values),
            count=n,
        )

    return UsageResponse(
        tenant_id=tenant.id,
        metrics=points,
        aggregation=agg,
    )


@router.get("/usage/models/{model_id}", response_model=UsageResponse)
def get_model_usage(
    model_id: str,
    metric: Optional[str] = Query(None),
    limit: int = Query(100, le=1000),
    tenant: Tenant = Depends(get_current_tenant),
    db: Session = Depends(get_db),
    _user: User = Depends(require_editor),
):
    """Métricas de un modelo específico."""
    query = (
        db.query(UsageMetric)
        .filter(
            UsageMetric.tenant_id == tenant.id,
            UsageMetric.model_id == model_id,
        )
    )
    if metric:
        query = query.filter(UsageMetric.metric_name == metric)
    items = query.order_by(UsageMetric.recorded_at.desc()).limit(limit).all()

    return UsageResponse(
        tenant_id=tenant.id,
        model_id=model_id,
        metrics=[
            UsagePointResponse(
                recorded_at=m.recorded_at,
                metric_name=m.metric_name,
                metric_value=m.metric_value,
                tags=m.tags,
            )
            for m in items
        ],
    )


# ── Costs ─────────────────────────────────────────────────────────────────


@router.get("/costs")
def get_costs(
    cost_type: Optional[str] = Query(None),
    period: str = Query("monthly"),
    tenant: Tenant = Depends(get_current_tenant),
    db: Session = Depends(get_db),
    _user: User = Depends(require_editor),
):
    """Desglose de costes."""
    query = db.query(CostRecord).filter(CostRecord.tenant_id == tenant.id)

    if cost_type:
        query = query.filter(CostRecord.cost_type == cost_type)

    if period == "monthly":
        now = datetime.now(timezone.utc)
        month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        query = query.filter(CostRecord.period_start >= month_start)

    total = query.with_entities(func.coalesce(func.sum(CostRecord.amount), 0)).scalar()
    records = query.order_by(CostRecord.period_start.desc()).limit(200).all()

    return {
        "tenant_id": tenant.id,
        "period": period,
        "total_cost_usd": round(float(total), 2),
        "records": [
            {
                "id": r.id,
                "cost_type": r.cost_type,
                "amount": r.amount,
                "currency": r.currency,
                "source": r.source,
                "model_id": r.model_id,
                "period_start": r.period_start.isoformat(),
                "period_end": r.period_end.isoformat(),
            }
            for r in records
        ],
    }


@router.get("/costs/forecast", response_model=CostForecastResponse)
def get_cost_forecast(
    tenant: Tenant = Depends(get_current_tenant),
    db: Session = Depends(get_db),
    calculator: CostCalculator = Depends(get_cost_calculator_dep),
    _user: User = Depends(require_editor),
):
    """Predicción de coste mensual vía regresión lineal."""
    projected = None
    trend = None

    start = datetime.now(timezone.utc) - timedelta(days=90)
    try:
        daily = (
            db.query(
                func.date_trunc("day", CostRecord.period_start).label("day"),
                func.sum(CostRecord.amount).label("total"),
            )
            .filter(
                CostRecord.tenant_id == tenant.id,
                CostRecord.period_start >= start,
            )
            .group_by(func.date_trunc("day", CostRecord.period_start))
            .order_by("day")
            .all()
        )
    except Exception:
        logger.warning("date_trunc no soportado por la BD, omitiendo forecast")
        daily = []

    if len(daily) >= 7:
        x = list(range(len(daily)))
        y = [float(d.total) for d in daily]
        n = len(x)
        slope = (n * sum(xi * yi for xi, yi in zip(x, y)) - sum(x) * sum(y)) / (
            n * sum(xi * xi for xi in x) - sum(x) * sum(x)
        ) if (n * sum(xi * xi for xi in x) - sum(x) * sum(x)) != 0 else 0
        intercept = (sum(y) - slope * sum(x)) / n

        days_in_month = (
            datetime.now(timezone.utc).replace(day=28) + timedelta(days=4)
        ).day
        today = datetime.now(timezone.utc).day
        remaining_days = days_in_month - today + 1
        projected = round(intercept + slope * (n + remaining_days), 2)
        trend = round(slope, 4)

    tenant_obj = (
        db.query(Tenant).filter(Tenant.id == tenant.id).first()
    )
    budget = tenant_obj.monthly_budget_usd if tenant_obj else None

    return CostForecastResponse(
        projected_month_end_usd=projected,
        budget_usd=budget,
        overrun_risk_pct=round((projected / budget) * 100, 1)
        if (projected and budget and budget > 0)
        else None,
        daily_trend_usd=trend,
    )


# ── Pricing (admin) ───────────────────────────────────────────────────────


@router.post("/pricing", response_model=ResourcePricingResponse)
def create_pricing(
    data: ResourcePricingCreate,
    db: Session = Depends(get_db),
    _user: User = Depends(require_admin),
):
    """Crear/actualizar precio. Desactiva el anterior si existe."""
    existing = (
        db.query(ResourcePricing)
        .filter(
            ResourcePricing.resource_category == data.resource_category,
            ResourcePricing.resource_id == data.resource_id,
            ResourcePricing.pricing_unit == data.pricing_unit,
            ResourcePricing.effective_to.is_(None),
        )
        .all()
    )
    for e in existing:
        e.effective_to = datetime.now(timezone.utc)

    pricing = ResourcePricing(
        resource_category=data.resource_category,
        resource_id=data.resource_id,
        pricing_unit=data.pricing_unit,
        price_per_unit=data.price_per_unit,
        currency=data.currency or "USD",
        extra_metadata=data.metadata,
    )
    db.add(pricing)
    db.commit()
    db.refresh(pricing)
    logger.info(
        "Precio actualizado: %s/%s/%s → %.4f",
        data.resource_category,
        data.resource_id,
        data.pricing_unit,
        data.price_per_unit,
    )
    return pricing


@router.get("/pricing", response_model=list[ResourcePricingResponse])
def list_pricing(
    category: Optional[str] = Query(None),
    db: Session = Depends(get_db),
    _user: User = Depends(require_editor),
):
    """Listar precios activos."""
    query = db.query(ResourcePricing).filter(ResourcePricing.effective_to.is_(None))
    if category:
        query = query.filter(ResourcePricing.resource_category == category)
    return query.order_by(
        ResourcePricing.resource_category, ResourcePricing.resource_id
    ).all()


@router.get("/pricing/history", response_model=ResourcePricingHistoryResponse)
def get_pricing_history(
    category: Optional[str] = Query(None),
    resource_id: Optional[str] = Query(None),
    db: Session = Depends(get_db),
    _user: User = Depends(require_admin),
):
    """Historial completo de cambios de precio."""
    query = db.query(ResourcePricing)
    if category:
        query = query.filter(ResourcePricing.resource_category == category)
    if resource_id:
        query = query.filter(ResourcePricing.resource_id == resource_id)
    items = query.order_by(
        ResourcePricing.resource_category,
        ResourcePricing.resource_id,
        ResourcePricing.effective_from.desc(),
    ).all()
    return ResourcePricingHistoryResponse(
        history=[ResourcePricingResponse.model_validate(i) for i in items]
    )


# ── Drift ─────────────────────────────────────────────────────────────────


@router.get("/drift/history/{model_id}", response_model=DriftHistoryResponse)
def get_drift_history(
    model_id: str,
    limit: int = Query(200, le=1000),
    tenant: Tenant = Depends(get_current_tenant),
    db: Session = Depends(get_db),
    _user: User = Depends(require_editor),
):
    """Serie temporal de drift para un modelo."""
    items = (
        db.query(DriftHistory)
        .filter(
            DriftHistory.model_id == model_id,
            DriftHistory.tenant_id == tenant.id,
        )
        .order_by(DriftHistory.recorded_at.desc())
        .limit(limit)
        .all()
    )
    return DriftHistoryResponse(
        model_id=model_id,
        drift_points=[
            DriftPointResponse(
                recorded_at=d.recorded_at,
                drift_type=d.drift_type,
                psi=(d.metrics or {}).get("psi"),
                ks=(d.metrics or {}).get("ks"),
                is_anomaly=d.is_anomaly,
            )
            for d in items
        ],
    )


# ── Alerts ────────────────────────────────────────────────────────────────


@router.post("/alerts", response_model=AlertResponse, status_code=201)
def create_alert(
    data: AlertCreate,
    tenant: Tenant = Depends(get_current_tenant),
    db: Session = Depends(get_db),
    _user: User = Depends(require_editor),
):
    """Crear alerta."""
    alert = AlertConfig(
        tenant_id=tenant.id,
        alert_type=data.alert_type,
        metric_name=data.metric_name,
        threshold=data.threshold,
        comparison=data.comparison,
        cooldown_min=data.cooldown_min,
        channels=data.channels,
    )
    db.add(alert)
    db.commit()
    db.refresh(alert)
    return alert


@router.get("/alerts", response_model=list[AlertResponse])
def list_alerts(
    tenant: Tenant = Depends(get_current_tenant),
    db: Session = Depends(get_db),
    _user: User = Depends(require_editor),
):
    """Listar alertas del tenant."""
    return (
        db.query(AlertConfig)
        .filter(AlertConfig.tenant_id == tenant.id)
        .order_by(AlertConfig.created_at.desc())
        .all()
    )


@router.patch("/alerts/{alert_id}", response_model=AlertResponse)
def update_alert(
    alert_id: str,
    data: AlertUpdate,
    tenant: Tenant = Depends(get_current_tenant),
    db: Session = Depends(get_db),
    _user: User = Depends(require_editor),
):
    """Actualizar alerta."""
    alert = (
        db.query(AlertConfig)
        .filter(
            AlertConfig.id == alert_id,
            AlertConfig.tenant_id == tenant.id,
        )
        .first()
    )
    if not alert:
        raise HTTPException(status_code=404, detail="Alerta no encontrada")

    update_data = data.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(alert, field, value)
    db.commit()
    db.refresh(alert)
    return alert


@router.delete("/alerts/{alert_id}", status_code=204)
def delete_alert(
    alert_id: str,
    tenant: Tenant = Depends(get_current_tenant),
    db: Session = Depends(get_db),
    _user: User = Depends(require_admin),
):
    """Eliminar alerta."""
    alert = (
        db.query(AlertConfig)
        .filter(
            AlertConfig.id == alert_id,
            AlertConfig.tenant_id == tenant.id,
        )
        .first()
    )
    if not alert:
        raise HTTPException(status_code=404, detail="Alerta no encontrada")
    db.delete(alert)
    db.commit()


# ── Model Health ──────────────────────────────────────────────────────────


@router.get("/models/{model_id}/health", response_model=HealthScoreResponse)
def get_model_health(
    model_id: str,
    tenant: Tenant = Depends(get_current_tenant),
    db: Session = Depends(get_db),
    _user: User = Depends(require_viewer),
):
    """Health Score compuesto (0-100)."""
    from app.models.ml_model import MLModel

    model = (
        db.query(MLModel)
        .filter(
            MLModel.id == model_id,
            MLModel.tenant_id == tenant.id,
        )
        .first()
    )
    if not model:
        raise HTTPException(status_code=404, detail="Modelo no encontrado")

    accuracy_score_val = 1.0
    drift_risk = 0.0
    latency_ratio = 1.0

    latest_drift = (
        db.query(DriftHistory)
        .filter(
            DriftHistory.model_id == model_id,
            DriftHistory.tenant_id == tenant.id,
        )
        .order_by(DriftHistory.recorded_at.desc())
        .first()
    )
    if latest_drift and latest_drift.metrics:
        drift_risk = min(latest_drift.metrics.get("psi", 0) / 0.2, 1.0)

    recent_metrics = (
        db.query(UsageMetric)
        .filter(
            UsageMetric.model_id == model_id,
            UsageMetric.tenant_id == tenant.id,
            UsageMetric.metric_name == "inference_latency_ms",
        )
        .order_by(UsageMetric.recorded_at.desc())
        .limit(100)
        .all()
    )
    if recent_metrics:
        vals = [m.metric_value for m in recent_metrics]
        p99 = sorted(vals)[int(len(vals) * 0.99)]
        latency_ratio = min(p99 / 1000.0, 1.0)

    if model.metrics_metadata:
        accuracy_score_val = min(
            model.metrics_metadata.get("f1", 1.0),
            model.metrics_metadata.get("accuracy", 1.0),
            1.0,
        )

    components = [
        HealthComponent(
            name="accuracy",
            score=round(accuracy_score_val * 100, 1),
            status="healthy" if accuracy_score_val >= 0.8 else "warning" if accuracy_score_val >= 0.6 else "critical",
        ),
        HealthComponent(
            name="drift",
            score=round((1 - drift_risk) * 100, 1),
            status="healthy" if drift_risk < 0.5 else "warning" if drift_risk < 0.8 else "critical",
        ),
        HealthComponent(
            name="latency",
            score=round((1 - latency_ratio) * 100, 1),
            status="healthy" if latency_ratio < 0.5 else "warning",
        ),
    ]

    health_score = round(
        0.4 * components[0].score
        + 0.3 * components[1].score
        + 0.2 * components[2].score
        + 0.1 * 100,
        1,
    )

    return HealthScoreResponse(
        model_id=model_id,
        health_score=min(health_score, 100),
        components=components,
    )


# ── Report ────────────────────────────────────────────────────────────────


@router.get("/tenant/{tenant_id}/report")
def get_tenant_report(
    tenant_id: str,
    format: str = Query("pdf", pattern="^(pdf|csv)$"),
    month: Optional[str] = Query(None, pattern=r"^\d{4}-\d{2}$"),
    db: Session = Depends(get_db),
    _user: User = Depends(require_admin),
):
    """Reporte mensual PDF/CSV."""
    from app.observability.reports import generate_cost_csv, generate_monthly_report_pdf
    from fastapi.responses import Response

    if not month:
        now = datetime.now(timezone.utc)
        month = f"{now.year}-{now.month:02d}"

    year_str, month_str = month.split("-")
    year, month_num = int(year_str), int(month_str)
    from_date = datetime(year, month_num, 1, tzinfo=timezone.utc)
    if month_num == 12:
        to_date = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
    else:
        to_date = datetime(year, month_num + 1, 1, tzinfo=timezone.utc)

    if format == "csv":
        content = generate_cost_csv(db, tenant_id, from_date, to_date)
        return Response(
            content=content,
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=cost_report_{month}.csv"},
        )
    else:
        content = generate_monthly_report_pdf(db, tenant_id, month)
        return Response(
            content=content,
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename=monthly_report_{month}.pdf"},
        )


# ── Unified Platform Observability Summary ────────────────────────────────────


@router.get("/platform-summary")
def get_platform_summary(
    tenant: Tenant = Depends(get_current_tenant),
    db: Session = Depends(get_db),
    _user: User = Depends(require_viewer),
):
    """
    Resumen integral de observabilidad de la plataforma para el tenant:
    salud de servicios (PostgreSQL, Redis/colas, Storage, MLflow),
    Model Serving Endpoints, Tracking/DVC/Storage, Runtime Environments,
    Pipelines, Governance y distribución de UsageMetric por dominio.
    """
    import urllib.request
    from sqlalchemy import text

    from app.core.config import settings
    from app.core.redis import create_sync_redis_client
    from app.models.api_key import APIKey
    from app.models.audit_log import AuditLog
    from app.models.dataset import Dataset
    from app.models.ml_model import MLModel
    from app.models.model_endpoint import ModelEndpoint
    from app.models.runtime_environment import RuntimeEnvironment
    from app.models.user_pipeline import UserPipeline
    from app.models.workspace import Workspace
    from app.services.storage_service import get_storage

    # 1. Live infrastructure health probes
    pg_status = "ok"
    try:
        db.execute(text("SELECT 1"))
    except Exception:
        pg_status = "error"

    redis_status = "unavailable"
    queue_depths: dict[str, int] = {}
    try:
        rclient = create_sync_redis_client()
        if rclient is not None and rclient.ping():
            redis_status = "ok"
            for q in ("cpu-light", "cpu-heavy", "gpu-train", "system-beat", "celery"):
                queue_depths[q] = int(rclient.llen(q) or 0)
    except Exception:
        redis_status = "degraded"

    storage_status = "ok"
    try:
        _ = get_storage()
    except Exception:
        storage_status = "error"

    mlflow_uri = settings.MLFLOW_TRACKING_URI or "local-file"
    mlflow_status = "ok"
    if mlflow_uri.startswith("http"):
        try:
            with urllib.request.urlopen(f"{mlflow_uri.rstrip('/')}/health", timeout=1.5) as resp:
                mlflow_status = "ok" if resp.status < 400 else "degraded"
        except Exception:
            mlflow_status = "degraded"

    # 2. Deployed Model Serving Endpoints
    endpoints = (
        db.query(ModelEndpoint)
        .filter(ModelEndpoint.tenant_id == tenant.id)
        .order_by(ModelEndpoint.request_count.desc())
        .all()
    )
    active_eps = sum(1 for e in endpoints if e.status == "active")
    paused_eps = len(endpoints) - active_eps
    total_reqs = sum(e.request_count or 0 for e in endpoints)
    total_errs = sum(e.error_count or 0 for e in endpoints)
    weighted_lat = (
        round(
            sum((e.avg_latency_ms or 0.0) * (e.request_count or 0) for e in endpoints) / total_reqs,
            2,
        )
        if total_reqs > 0
        else 0.0
    )
    error_rate_pct = round((total_errs / total_reqs) * 100.0, 2) if total_reqs > 0 else 0.0

    # 3. Tracking, Datasets & Storage footprint
    datasets = db.query(Dataset).filter(Dataset.tenant_id == tenant.id).all()
    dvc_tracked = sum(1 for d in datasets if d.is_dvc_tracked)
    total_bytes = sum(d.file_size_bytes or 0 for d in datasets)

    models = (
        db.query(MLModel)
        .filter(MLModel.tenant_id == tenant.id, MLModel.is_active.is_(True))
        .all()
    )
    models_by_stage: dict[str, int] = {"Production": 0, "Staging": 0, "Archived": 0}
    for mdl in models:
        st = mdl.stage or "Staging"
        models_by_stage[st] = models_by_stage.get(st, 0) + 1

    # 4. Compute, Environments, Pipelines & Governance
    env_count = (
        db.query(func.count(RuntimeEnvironment.id))
        .filter(RuntimeEnvironment.tenant_id == tenant.id)
        .scalar()
        or 0
    )
    pipe_count = (
        db.query(func.count(UserPipeline.id))
        .filter(UserPipeline.tenant_id == tenant.id)
        .scalar()
        or 0
    )
    ws_count = (
        db.query(func.count(Workspace.id))
        .filter(Workspace.tenant_id == tenant.id)
        .scalar()
        or 0
    )
    active_keys = (
        db.query(func.count(APIKey.id))
        .filter(APIKey.tenant_id == tenant.id, APIKey.revoked_at.is_(None))
        .scalar()
        or 0
    )
    since_24h = datetime.now(timezone.utc) - timedelta(hours=24)
    audit_24h = (
        db.query(func.count(AuditLog.id))
        .filter(AuditLog.tenant_id == tenant.id, AuditLog.created_at >= since_24h)
        .scalar()
        or 0
    )

    # 5. Usage metrics count by domain (serving, tabular, nlp, pipelines, llm)
    domain_rows = (
        db.query(UsageMetric.domain, func.count(UsageMetric.id))
        .filter(UsageMetric.tenant_id == tenant.id)
        .group_by(UsageMetric.domain)
        .all()
    )
    usage_by_domain = {str(dom or "unknown"): int(cnt) for dom, cnt in domain_rows}

    return {
        "tenant_id": tenant.id,
        "services_health": {
            "postgres": {"status": pg_status},
            "redis": {"status": redis_status, "queues": queue_depths},
            "storage": {"status": storage_status, "backend": settings.STORAGE_BACKEND},
            "mlflow": {"status": mlflow_status, "tracking_uri": mlflow_uri},
        },
        "serving": {
            "total_endpoints": len(endpoints),
            "active_endpoints": active_eps,
            "paused_endpoints": paused_eps,
            "total_requests": total_reqs,
            "total_errors": total_errs,
            "error_rate_pct": error_rate_pct,
            "avg_latency_ms": weighted_lat,
            "endpoints": [
                {
                    "id": e.id,
                    "name": e.name,
                    "slug": e.slug,
                    "status": e.status,
                    "traffic_split_pct": e.traffic_split_pct,
                    "request_count": e.request_count or 0,
                    "error_count": e.error_count or 0,
                    "avg_latency_ms": e.avg_latency_ms or 0.0,
                    "last_invoked_at": e.last_invoked_at.isoformat() if e.last_invoked_at else None,
                }
                for e in endpoints
            ],
        },
        "tracking_and_data": {
            "total_datasets": len(datasets),
            "dvc_tracked_datasets": dvc_tracked,
            "datasets_storage_bytes": total_bytes,
            "datasets_storage_mb": round(total_bytes / (1024 * 1024), 3),
            "total_models": len(models),
            "models_by_stage": models_by_stage,
        },
        "compute_and_governance": {
            "runtime_environments": int(env_count),
            "user_pipelines": int(pipe_count),
            "workspaces": int(ws_count),
            "active_api_keys": int(active_keys),
            "audit_events_24h": int(audit_24h),
        },
        "usage_by_domain": usage_by_domain,
    }
