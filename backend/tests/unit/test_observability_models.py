"""Tests unitarios para los modelos ORM de observabilidad."""

from datetime import datetime, timezone

import pytest

from app.observability.models import (
    UsageMetric,
    CostRecord,
    DriftHistory,
    AlertConfig,
    ResourcePricing,
)


class TestUsageMetric:
    def test_create_usage_metric(self):
        now = datetime.now(timezone.utc)
        metric = UsageMetric(
            tenant_id="tenant-1",
            model_id="model-1",
            domain="tabular",
            metric_name="inference_latency_ms",
            metric_value=150.5,
            tags={"algorithm": "random_forest"},
            recorded_at=now,
        )
        assert metric.tenant_id == "tenant-1"
        assert metric.model_id == "model-1"
        assert metric.domain == "tabular"
        assert metric.metric_name == "inference_latency_ms"
        assert metric.metric_value == 150.5
        assert metric.tags == {"algorithm": "random_forest"}
        assert metric.recorded_at == now
        assert metric.id is None

    def test_usage_metric_table_name(self):
        assert UsageMetric.__tablename__ == "usage_metric"


class TestCostRecord:
    def test_create_cost_record(self):
        now = datetime.now(timezone.utc)
        record = CostRecord(
            tenant_id="tenant-1",
            model_id="model-1",
            cost_type="inference",
            amount=0.05,
            currency="USD",
            source="prediction-123",
            period_start=now,
            period_end=now,
        )
        assert record.tenant_id == "tenant-1"
        assert record.amount == 0.05
        assert record.currency == "USD"

    def test_cost_record_table_name(self):
        assert CostRecord.__tablename__ == "cost_record"


class TestDriftHistory:
    def test_create_drift_history(self):
        now = datetime.now(timezone.utc)
        drift = DriftHistory(
            model_id="model-1",
            tenant_id="tenant-1",
            drift_type="data_drift",
            metrics={"psi": 0.15, "ks": 0.03},
            is_anomaly=False,
            recorded_at=now,
        )
        assert drift.model_id == "model-1"
        assert drift.drift_type == "data_drift"
        assert drift.metrics["psi"] == 0.15
        assert drift.is_anomaly is False

    def test_drift_history_table_name(self):
        assert DriftHistory.__tablename__ == "drift_history"


class TestAlertConfig:
    def test_create_alert_config(self):
        alert = AlertConfig(
            tenant_id="tenant-1",
            alert_type="cost_spike",
            metric_name="inference_cost_usd",
            threshold=2.0,
            comparison="pct_change",
            channels=[{"type": "webhook", "url": "https://hooks.slack.com/..."}],
            enabled=True,
        )
        assert alert.alert_type == "cost_spike"
        assert alert.enabled is True

    def test_alert_config_table_name(self):
        assert AlertConfig.__tablename__ == "alert_config"


class TestResourcePricing:
    def test_create_resource_pricing(self):
        price = ResourcePricing(
            resource_category="compute_gpu",
            resource_id="A100",
            pricing_unit="per_hour",
            price_per_unit=3.50,
            currency="USD",
        )
        assert price.resource_category == "compute_gpu"
        assert price.resource_id == "A100"
        assert price.price_per_unit == 3.50
        assert price.currency == "USD"
        assert price.effective_to is None

    def test_resource_pricing_table_name(self):
        assert ResourcePricing.__tablename__ == "resource_pricing"
