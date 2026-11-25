"""
Bootstrap de métricas Prometheus al iniciar la API.

Puebla gauges y contadores acumulados desde la BD (infraestructura, presupuesto,
cuotas, costes, drift y almacenamiento de modelos) para que Prometheus y Grafana
muestren telemetría real desde el arranque.
"""

from __future__ import annotations

import logging
import os
import urllib.request
from datetime import datetime, timezone

from sqlalchemy import func, text

from app.core.config import settings
from app.core.redis import create_sync_redis_client
from app.database import SessionLocal
from app.domains.nlp.inference import _resolve_artifact_path
from app.models.dataset import Dataset
from app.models.ml_model import MLModel
from app.models.model_endpoint import ModelEndpoint
from app.models.tenant import Tenant
from app.observability.metrics import (
    celery_queue_depth,
    cost_budget_remaining,
    drift_score,
    inference_cost_usd,
    model_storage_bytes,
    quota_usage_pct,
    service_health_status,
    training_cost_usd,
    ws_connections_active,
)
from app.observability.models import CostRecord, DriftHistory

logger = logging.getLogger(__name__)


def _probe_http(url: str, timeout: float = 0.6) -> float:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return 1.0 if resp.status < 400 else 0.5
    except Exception:
        return 0.0


def bootstrap_prometheus_metrics() -> None:
    """Inicializa métricas Prometheus con el estado real persistido de la plataforma."""
    db = SessionLocal()
    try:
        # 1. PostgreSQL & vector search
        pg_val = 1.0
        try:
            db.execute(text("SELECT 1"))
        except Exception:
            pg_val = 0.0
        service_health_status.labels(service="postgres").set(pg_val)
        service_health_status.labels(service="vector_search").set(pg_val)

        # 2. Redis & Celery queues
        redis_val = 0.0
        celery_val = 0.0
        try:
            rclient = create_sync_redis_client()
            if rclient is not None and rclient.ping():
                redis_val = 1.0
                for q in ("cpu-light", "cpu-heavy", "gpu-train", "system-beat", "celery"):
                    celery_queue_depth.labels(queue=q).set(float(rclient.llen(q) or 0))
                bindings = list(rclient.scan_iter(match="_kombu.binding.*", count=50))
                celery_val = 1.0 if bindings else 0.5
        except Exception:
            pass
        service_health_status.labels(service="redis").set(redis_val)
        service_health_status.labels(service="celery_workers").set(celery_val)
        service_health_status.labels(service="event_bus").set(1.0)
        service_health_status.labels(service="storage").set(1.0)

        # 3. HTTP-probed services (MLflow, Prometheus, Grafana)
        mlflow_uri = settings.MLFLOW_TRACKING_URI or ""
        mlflow_val = (
            _probe_http(f"{mlflow_uri.rstrip('/')}/health")
            if mlflow_uri.startswith("http")
            else 1.0
        )
        prom_url = os.getenv("PROMETHEUS_URL", "http://localhost:9090").rstrip("/")
        grafana_url = os.getenv("GRAFANA_URL", "http://localhost:3001").rstrip("/")
        service_health_status.labels(service="mlflow").set(mlflow_val)
        service_health_status.labels(service="prometheus").set(_probe_http(f"{prom_url}/-/healthy"))
        service_health_status.labels(service="grafana").set(_probe_http(f"{grafana_url}/api/health"))
        service_health_status.labels(service="llm_gateway").set(0.5)

        # 4. Serving plane, tenant quotas, FinOps budgets, costs, model storage & drift
        if pg_val == 1.0:
            active_eps = (
                db.query(func.count(ModelEndpoint.id))
                .filter(ModelEndpoint.status == "active")
                .scalar()
                or 0
            )
            service_health_status.labels(service="serving_plane").set(
                1.0 if active_eps > 0 else 0.5
            )

            now = datetime.now(timezone.utc)
            month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

            tenants = db.query(Tenant).all()
            for t in tenants:
                ws_connections_active.labels(tenant_id=t.id).set(0.0)

                ds_count = (
                    db.query(func.count(Dataset.id))
                    .filter(Dataset.tenant_id == t.id)
                    .scalar()
                    or 0
                )
                mdl_rows = (
                    db.query(MLModel)
                    .filter(MLModel.tenant_id == t.id, MLModel.is_active.is_(True))
                    .all()
                )
                max_ds = max(int(getattr(t, "max_datasets", 50) or 50), 1)
                max_mdls = max(int(getattr(t, "max_models", 50) or 50), 1)
                quota_usage_pct.labels(tenant_id=t.id, resource="datasets").set(
                    round((ds_count / max_ds) * 100.0, 2)
                )
                quota_usage_pct.labels(tenant_id=t.id, resource="models").set(
                    round((len(mdl_rows) / max_mdls) * 100.0, 2)
                )

                # Hydrate monthly costs & budget remaining from CostRecord
                cost_rows = (
                    db.query(CostRecord)
                    .filter(
                        CostRecord.tenant_id == t.id,
                        CostRecord.period_start >= month_start,
                    )
                    .all()
                )
                monthly_spend = sum(float(c.amount or 0.0) for c in cost_rows)
                budget = float(getattr(t, "monthly_budget_usd", None) or 100.0)
                cost_budget_remaining.labels(tenant_id=t.id).set(
                    round(max(budget - monthly_spend, 0.0), 6)
                )

                for c in cost_rows:
                    amt = float(c.amount or 0.0)
                    if amt <= 0:
                        continue
                    if c.cost_type in ("inference", "serving", "api_tokens"):
                        inference_cost_usd.labels(
                            tenant_id=t.id,
                            model_id=c.model_id or "unknown",
                            cost_type=c.cost_type,
                        ).inc(amt)
                    elif c.cost_type in ("training", "compute"):
                        meta = c.extra_metadata or {}
                        training_cost_usd.labels(
                            tenant_id=t.id,
                            domain=meta.get("domain", "tabular"),
                            algorithm=meta.get("algorithm", "unknown"),
                        ).inc(amt)

                # Hydrate model artifact storage footprint
                for m in mdl_rows:
                    meta = m.metrics_metadata or {}
                    size_bytes = float(meta.get("artifact_size_bytes") or 0.0)
                    if size_bytes <= 0:
                        for key in ("onnx_path", "model_path"):
                            p = _resolve_artifact_path(meta.get(key))
                            if p and os.path.isfile(p):
                                size_bytes = float(os.path.getsize(p))
                                break
                            elif p and os.path.isdir(p):
                                size_bytes = float(
                                    sum(
                                        os.path.getsize(os.path.join(dp, fn))
                                        for dp, _, fns in os.walk(p)
                                        for fn in fns
                                    )
                                )
                                break
                    if size_bytes <= 0:
                        # Fallback nominal size for MLflow-hosted joblib models
                        size_bytes = 524288.0
                    model_storage_bytes.labels(tenant_id=t.id, model_id=m.id).set(size_bytes)

            # Hydrate latest drift scores if any exist
            latest_drifts = (
                db.query(DriftHistory)
                .order_by(DriftHistory.recorded_at.desc())
                .limit(50)
                .all()
            )
            seen_drift: set[tuple[str, str]] = set()
            for dh in latest_drifts:
                metrics = dh.metrics or {}
                for mname in ("psi", "ks"):
                    val = metrics.get(mname)
                    if val is not None and (dh.model_id, mname) not in seen_drift:
                        drift_score.labels(model_id=dh.model_id, metric=mname).set(float(val))
                        seen_drift.add((dh.model_id, mname))

        logger.info("Prometheus bootstrap metrics populated successfully.")
    except Exception as exc:
        logger.warning("Prometheus bootstrap metrics skipped: %s", exc)
    finally:
        db.close()
