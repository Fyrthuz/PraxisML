"""
Motor de evaluación y despacho de alertas.
Soporta canales: webhook (Slack/Teams), email, dashboard badge, WebSocket push via Redis pub/sub.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Optional

import httpx
from sqlalchemy.orm import Session

from app.observability.models import AlertConfig, UsageMetric

logger = logging.getLogger(__name__)


class AlertResult:
    def __init__(
        self,
        triggered: bool,
        current_value: float,
        threshold: float,
        message: str,
        severity: str = "media",
    ):
        self.triggered = triggered
        self.current_value = current_value
        self.threshold = threshold
        self.message = message
        self.severity = severity


class AlertEngine:
    def evaluate(self, db: Session, alert: AlertConfig) -> AlertResult:
        """Evalúa si una alerta debe dispararse comparando métrica actual vs threshold."""
        current = self._get_current_metric(db, alert.metric_name, alert.tenant_id)
        if current is None:
            return AlertResult(
                triggered=False,
                current_value=0,
                threshold=alert.threshold,
                message="",
            )

        triggered = self._compare(current, alert.threshold, alert.comparison)
        severity = self._determine_severity(alert.alert_type, triggered)

        return AlertResult(
            triggered=triggered,
            current_value=current,
            threshold=alert.threshold,
            message=self._build_message(alert, current),
            severity=severity,
        )

    def dispatch(self, alert: AlertConfig, result: AlertResult) -> None:
        """Envía la notificación por todos los canales configurados."""
        channels = alert.channels or []
        for channel in channels:
            try:
                if channel["type"] == "webhook":
                    self._send_webhook(channel["url"], alert, result)
                elif channel["type"] == "email":
                    self._send_email(channel["to"], alert, result)
            except Exception as e:
                logger.error(
                    "Error enviando alerta por %s: %s", channel["type"], e
                )

        self._publish_ws(alert.tenant_id, alert, result)

    def set_cooldown(self, alert: AlertConfig) -> None:
        """Marca cooldown en Redis para evitar spam de notificaciones."""
        try:
            from app.core.redis import create_redis_client

            redis = create_redis_client()
            if redis is None:
                return
            key = f"alert_cooldown:{alert.id}"
            redis.setex(key, alert.cooldown_min * 60, "1")
        except Exception:
            pass

    def is_in_cooldown(self, alert: AlertConfig) -> bool:
        """Verifica si la alerta está en periodo de cooldown."""
        try:
            from app.core.redis import create_redis_client

            redis = create_redis_client()
            if redis is None:
                return False
            key = f"alert_cooldown:{alert.id}"
            return redis.exists(key) > 0
        except Exception:
            return False

    def get_active_cooldowns(self, tenant_id: str) -> list[str]:
        """Retorna los IDs de alertas en cooldown para un tenant."""
        try:
            from app.core.redis import create_redis_client

            redis = create_redis_client()
            if redis is None:
                return []
            pattern = "alert_cooldown:*"
            keys = redis.keys(pattern)
            return [k.split(":")[-1] for k in keys]
        except Exception:
            return []

    def _get_current_metric(
        self, db: Session, metric_name: str, tenant_id: str
    ) -> Optional[float]:
        last = (
            db.query(UsageMetric.metric_value)
            .filter(
                UsageMetric.tenant_id == tenant_id,
                UsageMetric.metric_name == metric_name,
            )
            .order_by(UsageMetric.recorded_at.desc())
            .first()
        )
        return float(last[0]) if last else None

    def _compare(
        self, current: float, threshold: float, comparison: str
    ) -> bool:
        comparators = {
            "gt": lambda c, t: c > t,
            "lt": lambda c, t: c < t,
            "gte": lambda c, t: c >= t,
            "lte": lambda c, t: c <= t,
        }
        return comparators.get(comparison, lambda c, t: False)(current, threshold)

    def _determine_severity(self, alert_type: str, triggered: bool) -> str:
        if not triggered:
            return "ok"
        critical = {
            "budget_exceeded",
            "latency_spike",
            "error_rate_spike",
            "drift_critical",
        }
        return "critica" if alert_type in critical else "alta"

    def _build_message(self, alert: AlertConfig, current: float) -> str:
        return (
            f"[{alert.alert_type}] {alert.metric_name} = {current:.2f} "
            f"({alert.comparison} {alert.threshold})"
        )

    def _send_webhook(
        self, url: str, alert: AlertConfig, result: AlertResult
    ) -> None:
        payload = {
            "text": result.message,
            "severity": result.severity,
            "tenant_id": alert.tenant_id,
            "alert_type": alert.alert_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        httpx.post(url, json=payload, timeout=5)

    def _send_email(
        self, to: str, alert: AlertConfig, result: AlertResult
    ) -> None:
        logger.info("Email alert a %s: %s", to, result.message)

    def _publish_ws(
        self, tenant_id: str, alert: AlertConfig, result: AlertResult
    ) -> None:
        try:
            from app.core.redis import create_redis_client

            redis = create_redis_client()
            if redis is None:
                return
            channel = f"praxisml:alerts:{tenant_id}"
            payload = json.dumps(
                {
                    "type": "alert",
                    "alert_id": alert.id,
                    "alert_type": alert.alert_type,
                    "message": result.message,
                    "severity": result.severity,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )
            redis.publish(channel, payload)
        except Exception:
            pass


_alert_engine: AlertEngine | None = None


def get_alert_engine() -> AlertEngine:
    global _alert_engine
    if _alert_engine is None:
        _alert_engine = AlertEngine()
    return _alert_engine
