"""
Tareas periódicas para observabilidad.
Ejecutadas por Celery Beat según schedule en celery_app.py.
"""

import logging
from datetime import datetime, timezone, timedelta

from app.database import SessionLocal
from app.observability.models import DriftHistory, AlertConfig
from app.observability.metrics import drift_score
from app.observability.alert_engine import get_alert_engine
from app.worker.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(name="app.worker.tasks.scheduled_checks.scheduled_drift_check")
def scheduled_drift_check():
    """Itera modelos en Production, calcula drift, persiste en drift_history."""
    db = SessionLocal()
    try:
        from app.models.ml_model import MLModel

        production_models = (
            db.query(MLModel)
            .filter(MLModel.stage == "Production", MLModel.is_active)
            .all()
        )
        for model in production_models:
            try:
                from app.api.routes.v1.drift import calculate_drift_for_model

                result = calculate_drift_for_model(model.id, db)
                if result:
                    history = DriftHistory(
                        model_id=model.id,
                        tenant_id=model.tenant_id,
                        drift_type="data_drift",
                        metrics={
                            "psi": result.get("psi", 0),
                            "ks": result.get("ks", 0),
                        },
                        is_anomaly=result.get("psi", 0) > 0.2,
                    )
                    db.add(history)
                    drift_score.labels(
                        model_id=model.id, metric="psi"
                    ).set(result.get("psi", 0))
                    drift_score.labels(
                        model_id=model.id, metric="ks"
                    ).set(result.get("ks", 0))
                    db.commit()
                    logger.info(
                        "Drift check OK para modelo %s: PSI=%.4f",
                        model.id,
                        result.get("psi", 0),
                    )
            except Exception as e:
                logger.error(
                    "Error en drift check para modelo %s: %s", model.id, e
                )
                db.rollback()
    finally:
        db.close()


@celery_app.task(name="app.worker.tasks.scheduled_checks.evaluate_alerts")
def evaluate_alerts():
    """Evalúa todas las alertas activas contra métricas actuales."""
    db = SessionLocal()
    try:
        alerts = (
            db.query(AlertConfig)
            .filter(AlertConfig.enabled)
            .all()
        )
        engine = get_alert_engine()
        for alert in alerts:
            try:
                if engine.is_in_cooldown(alert):
                    continue
                result = engine.evaluate(db, alert)
                if result.triggered:
                    engine.dispatch(alert, result)
                    engine.set_cooldown(alert)
                    logger.info(
                        "Alerta disparada: %s [%s]",
                        alert.alert_type,
                        alert.tenant_id,
                    )
            except Exception as e:
                logger.error(
                    "Error evaluando alerta %s: %s", alert.id, e
                )
    finally:
        db.close()


@celery_app.task(name="app.worker.tasks.scheduled_checks.check_zombie_models")
def check_zombie_models():
    """Detecta modelos en Production sin inferencias en los últimos 7 días."""
    db = SessionLocal()
    try:
        from sqlalchemy import func
        from app.models.ml_model import MLModel
        from app.models.prediction import Prediction

        cutoff = datetime.now(timezone.utc) - timedelta(days=7)
        prod_models = (
            db.query(MLModel)
            .filter(MLModel.stage == "Production", MLModel.is_active)
            .all()
        )
        for model in prod_models:
            last_pred = (
                db.query(func.max(Prediction.created_at))
                .filter(Prediction.model_id == model.id)
                .scalar()
            )
            last_activity = last_pred or model.promoted_at or model.created_at
            if last_activity and last_activity.tzinfo is None:
                last_activity = last_activity.replace(tzinfo=timezone.utc)
            if last_activity and last_activity < cutoff:
                logger.warning(
                    "Modelo zombie detectado: %s (tenant=%s, last_activity=%s)",
                    model.id,
                    model.tenant_id,
                    last_activity.isoformat(),
                )
    finally:
        db.close()


@celery_app.task(name="app.worker.tasks.scheduled_checks.check_budget_thresholds")
def check_budget_thresholds():
    """Por cada tenant con budget, si coste mensual > budget_alert_pct%, registra advertencia."""
    from app.models.tenant import Tenant
    from app.observability.cost_calculator import get_cost_calculator

    db = SessionLocal()
    try:
        calculator = get_cost_calculator()
        tenants = (
            db.query(Tenant)
            .filter(Tenant.monthly_budget_usd.isnot(None))
            .all()
        )
        for tenant in tenants:
            spent = calculator.get_monthly_cost(db, tenant.id)
            pct = (
                (spent / tenant.monthly_budget_usd) * 100
                if tenant.monthly_budget_usd
                else 0
            )
            if pct >= (tenant.budget_alert_pct or 80):
                logger.warning(
                    "Budget alert: tenant=%s, spent=%.2f, budget=%.2f, pct=%.1f%%",
                    tenant.id,
                    spent,
                    tenant.monthly_budget_usd,
                    pct,
                )
    finally:
        db.close()
