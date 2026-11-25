"""
UsageCollector — Singleton que captura métricas de uso,
las persiste en usage_metrics y las expone a Prometheus.
"""

import logging
from typing import Optional

from sqlalchemy.orm import Session

from app.observability import metrics as m
from app.observability.models import UsageMetric

logger = logging.getLogger(__name__)


class UsageCollector:
    def record_inference(
        self,
        db: Session,
        tenant_id: str,
        model_id: str,
        domain: str,
        latency_ms: float,
        tokens_input: int = 0,
        tokens_output: int = 0,
        success: bool = True,
        tags: Optional[dict] = None,
    ) -> None:
        metric = UsageMetric(
            tenant_id=tenant_id,
            model_id=model_id,
            domain=domain,
            metric_name="inference_latency_ms",
            metric_value=latency_ms,
            tags=tags or {},
        )
        db.add(metric)

        algo = (tags or {}).get("algorithm", "unknown")
        m.inference_latency_ms.labels(
            tenant_id=tenant_id, domain=domain, algorithm=algo, success=str(success).lower()
        ).observe(latency_ms)

        m.predictions_total.labels(
            tenant_id=tenant_id, model_id=model_id, status="success" if success else "error"
        ).inc()

        if tokens_input:
            m.inference_tokens_total.labels(
                tenant_id=tenant_id, model_id=model_id, direction="input"
            ).inc(tokens_input)
        if tokens_output:
            m.inference_tokens_total.labels(
                tenant_id=tenant_id, model_id=model_id, direction="output"
            ).inc(tokens_output)

    def record_training(
        self,
        db: Session,
        tenant_id: str,
        model_id: str,
        domain: str,
        duration_s: float,
        gpu_hours: float = 0,
        cpu_hours: float = 0,
        metrics: Optional[dict] = None,
        tags: Optional[dict] = None,
    ) -> None:
        metric = UsageMetric(
            tenant_id=tenant_id,
            model_id=model_id,
            domain=domain,
            metric_name="training_duration_s",
            metric_value=duration_s,
            tags=tags or {},
        )
        db.add(metric)

        algo = (tags or {}).get("algorithm", "unknown")
        m.training_duration_seconds.labels(
            tenant_id=tenant_id, domain=domain, algorithm=algo
        ).observe(duration_s)

    def record_storage(
        self,
        tenant_id: str,
        model_id: str,
        bytes_count: float,
        storage_type: str = "model",
    ) -> None:
        m.model_storage_bytes.labels(
            tenant_id=tenant_id, model_id=model_id
        ).set(bytes_count)

    def record_ws_connection(self, tenant_id: str, delta: int = 1) -> None:
        m.ws_connections_active.labels(tenant_id=tenant_id).inc(delta)

    def record_serving_invocation(
        self,
        db: Session,
        tenant_id: str,
        model_id: str,
        endpoint_slug: str,
        variant: str,
        runtime: str,
        latency_ms: float,
        success: bool = True,
        input_count: int = 1,
    ) -> None:
        """Registra una invocación de Model Endpoint en usage_metrics y Prometheus."""
        self.record_inference(
            db=db,
            tenant_id=tenant_id,
            model_id=model_id,
            domain="serving",
            latency_ms=latency_ms,
            success=success,
            tags={
                "endpoint_slug": endpoint_slug,
                "variant": variant,
                "runtime": runtime,
                "input_count": input_count,
                "algorithm": runtime,
            },
        )
        m.serving_requests_total.labels(
            tenant_id=tenant_id,
            endpoint_slug=endpoint_slug,
            variant=variant,
            status="success" if success else "error",
        ).inc()
        m.serving_latency_ms.labels(
            tenant_id=tenant_id,
            endpoint_slug=endpoint_slug,
            runtime=runtime,
        ).observe(latency_ms)

    def record_pipeline_run(
        self,
        db: Session,
        tenant_id: str,
        pipeline_id: str,
        engine: str,
        duration_s: float,
        success: bool = True,
        tags: Optional[dict] = None,
    ) -> None:
        """Registra una ejecución de UserPipeline en usage_metrics y Prometheus."""
        metric = UsageMetric(
            tenant_id=tenant_id,
            model_id=None,
            domain="pipelines",
            metric_name="pipeline_duration_s",
            metric_value=duration_s,
            tags={"pipeline_id": pipeline_id, "engine": engine, "success": success, **(tags or {})},
        )
        db.add(metric)
        m.pipeline_runs_total.labels(
            tenant_id=tenant_id,
            engine=engine,
            status="success" if success else "error",
        ).inc()
        m.pipeline_duration_seconds.labels(
            tenant_id=tenant_id,
            engine=engine,
        ).observe(duration_s)


_usage_collector: UsageCollector | None = None


def get_usage_collector() -> UsageCollector:
    global _usage_collector
    if _usage_collector is None:
        _usage_collector = UsageCollector()
    return _usage_collector
