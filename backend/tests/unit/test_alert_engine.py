"""Tests unitarios para AlertEngine."""

from unittest.mock import MagicMock, patch

import pytest

from app.observability.alert_engine import AlertEngine, AlertResult, get_alert_engine
from app.observability.models import AlertConfig


class TestAlertResult:
    def test_create_result(self):
        r = AlertResult(
            triggered=True,
            current_value=150.0,
            threshold=100.0,
            message="Latencia alta: 150 > 100",
            severity="alta",
        )
        assert r.triggered is True
        assert r.current_value == 150.0
        assert r.message == "Latencia alta: 150 > 100"


class TestAlertEngine:
    def setup_method(self):
        self.engine = AlertEngine()
        self.db = MagicMock()

    def test_evaluate_metric_found_and_triggered(self):
        # Simula una métrica actual retornada por DB
        self.db.query.return_value.filter.return_value.order_by.return_value.first.return_value = (150.0,)

        alert = AlertConfig(
            tenant_id="t1",
            alert_type="latency_degradation",
            metric_name="inference_latency_ms",
            threshold=100.0,
            comparison="gt",
            channels=[],
        )
        result = self.engine.evaluate(self.db, alert)
        assert result.triggered is True
        assert result.current_value == 150.0

    def test_evaluate_metric_found_not_triggered(self):
        self.db.query.return_value.filter.return_value.order_by.return_value.first.return_value = (50.0,)

        alert = AlertConfig(
            tenant_id="t1",
            alert_type="latency_degradation",
            metric_name="inference_latency_ms",
            threshold=100.0,
            comparison="gt",
            channels=[],
        )
        result = self.engine.evaluate(self.db, alert)
        assert result.triggered is False

    def test_evaluate_metric_not_found(self):
        self.db.query.return_value.filter.return_value.order_by.return_value.first.return_value = None

        alert = AlertConfig(
            tenant_id="t1",
            alert_type="cost_spike",
            metric_name="inference_cost_usd",
            threshold=2.0,
            comparison="pct_change",
            channels=[],
        )
        result = self.engine.evaluate(self.db, alert)
        assert result.triggered is False

    def test_compare_gt(self):
        assert self.engine._compare(150, 100, "gt") is True
        assert self.engine._compare(50, 100, "gt") is False

    def test_compare_lt(self):
        assert self.engine._compare(50, 100, "lt") is True
        assert self.engine._compare(150, 100, "lt") is False

    def test_compare_gte(self):
        assert self.engine._compare(100, 100, "gte") is True
        assert self.engine._compare(99, 100, "gte") is False

    def test_compare_lte(self):
        assert self.engine._compare(100, 100, "lte") is True
        assert self.engine._compare(101, 100, "lte") is False

    def test_compare_invalid_operator(self):
        assert self.engine._compare(100, 100, "invalid") is False

    def test_determine_severity_not_triggered(self):
        assert self.engine._determine_severity("cost_spike", False) == "ok"

    def test_determine_severity_critical(self):
        assert self.engine._determine_severity("budget_exceeded", True) == "critica"
        assert self.engine._determine_severity("latency_spike", True) == "critica"
        assert self.engine._determine_severity("drift_critical", True) == "critica"

    def test_determine_severity_alta(self):
        assert self.engine._determine_severity("cost_spike", True) == "alta"

    def test_build_message(self):
        alert = AlertConfig(
            tenant_id="t1",
            alert_type="cost_spike",
            metric_name="inference_cost_usd",
            threshold=2.0,
            comparison="gt",
            channels=[],
        )
        msg = self.engine._build_message(alert, 3.5)
        assert "cost_spike" in msg
        assert "3.50" in msg

    def test_dispatch_webhook(self):
        with patch("httpx.post") as mock_post:
            alert = AlertConfig(
                tenant_id="t1",
                alert_type="test",
                metric_name="test_metric",
                threshold=10,
                comparison="gt",
                channels=[{"type": "webhook", "url": "https://hooks.example.com"}],
            )
            result = AlertResult(True, 15, 10, "Test alert", "alta")
            self.engine.dispatch(alert, result)
            mock_post.assert_called_once()

    def test_cooldown_roundtrip(self):
        with patch("app.core.redis.create_sync_redis_client") as mock_redis_factory:
            mock_redis = MagicMock()
            mock_redis_factory.return_value = mock_redis
            mock_redis.exists.return_value = 1

            alert = AlertConfig(
                tenant_id="t1",
                alert_type="test",
                metric_name="test",
                threshold=10,
                comparison="gt",
                channels=[],
                cooldown_min=60,
            )

            # Redis exists → cooldown activo
            assert self.engine.is_in_cooldown(alert) is True

            # Redis no existe → cooldown inactivo
            mock_redis.exists.return_value = 0
            assert self.engine.is_in_cooldown(alert) is False

            # set_cooldown llama a setex
            self.engine.set_cooldown(alert)
            mock_redis.setex.assert_called_once()

    def test_singleton(self):
        e1 = get_alert_engine()
        e2 = get_alert_engine()
        assert e1 is e2
        assert isinstance(e1, AlertEngine)
