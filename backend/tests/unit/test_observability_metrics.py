"""Tests unitarios para las métricas Prometheus de observabilidad."""

from prometheus_client import Counter, Gauge, Histogram


# Mapeo de nombre Prometheus → variable en módulo metrics
METRIC_VARIABLES = {
    "praxisml_inference_latency_ms": "inference_latency_ms",
    "praxisml_inference_tokens_total": "inference_tokens_total",
    "praxisml_inference_cost_usd": "inference_cost_usd",
    "praxisml_training_duration_seconds": "training_duration_seconds",
    "praxisml_training_cost_usd": "training_cost_usd",
    "praxisml_model_storage_bytes": "model_storage_bytes",
    "praxisml_ws_connections_active": "ws_connections_active",
    "praxisml_drift_score": "drift_score",
    "praxisml_quota_usage_pct": "quota_usage_pct",
    "praxisml_cost_budget_remaining": "cost_budget_remaining",
    "praxisml_predictions_total": "predictions_total",
    "praxisml_errors_total": "errors_total",
}


class TestMetricsRegistered:
    def test_all_metrics_variables_exist(self):
        from app.observability import metrics as m

        for metric_name, var_name in METRIC_VARIABLES.items():
            assert hasattr(m, var_name), f"Falta variable '{var_name}' para métrica {metric_name}"

    def test_metrics_are_prometheus_types(self):
        from app.observability import metrics as m

        assert isinstance(m.inference_latency_ms, Histogram)
        assert isinstance(m.inference_tokens_total, Counter)
        assert isinstance(m.inference_cost_usd, Counter)
        assert isinstance(m.training_duration_seconds, Histogram)
        assert isinstance(m.training_cost_usd, Counter)
        assert isinstance(m.model_storage_bytes, Gauge)
        assert isinstance(m.ws_connections_active, Gauge)
        assert isinstance(m.drift_score, Gauge)
        assert isinstance(m.quota_usage_pct, Gauge)
        assert isinstance(m.cost_budget_remaining, Gauge)
        assert isinstance(m.predictions_total, Counter)
        assert isinstance(m.errors_total, Counter)

    def test_histogram_observe_and_read(self):
        from app.observability import metrics as m

        inf = m.inference_latency_ms
        inf.labels(tenant_id="t", domain="nlp", algorithm="bert", success="true").observe(10)
        inf.labels(tenant_id="t", domain="nlp", algorithm="bert", success="true").observe(5000)

        train = m.training_duration_seconds
        train.labels(tenant_id="t", domain="tabular", algorithm="rf").observe(600)
        train.labels(tenant_id="t", domain="tabular", algorithm="xgboost").observe(1800)

    def test_counter_can_be_incremented(self):
        from app.observability import metrics as m

        m.predictions_total.labels(tenant_id="test", model_id="m1", status="success").inc()
        m.predictions_total.labels(tenant_id="test", model_id="m1", status="error").inc(2)

    def test_gauge_can_be_set(self):
        from app.observability import metrics as m

        m.model_storage_bytes.labels(tenant_id="test", model_id="m1").set(1024)
        m.drift_score.labels(model_id="m1", metric="psi").set(0.15)
        m.ws_connections_active.labels(tenant_id="test").inc()
        m.ws_connections_active.labels(tenant_id="test").dec()
        m.quota_usage_pct.labels(tenant_id="test", resource="datasets").set(50)
        m.cost_budget_remaining.labels(tenant_id="test").set(100.0)

    def test_histogram_can_observe(self):
        from app.observability import metrics as m

        m.inference_latency_ms.labels(
            tenant_id="test", domain="tabular", algorithm="rf", success="true"
        ).observe(150)
        m.training_duration_seconds.labels(
            tenant_id="test", domain="tabular", algorithm="rf"
        ).observe(300)
