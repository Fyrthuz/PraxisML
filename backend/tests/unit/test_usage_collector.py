"""Tests unitarios para UsageCollector."""

from unittest.mock import MagicMock, patch

import pytest

from app.observability.usage_collector import UsageCollector, get_usage_collector


class TestUsageCollector:
    def setup_method(self):
        self.collector = UsageCollector()
        self.db = MagicMock()
        self.db.add = MagicMock()

    def test_record_inference_basic(self):
        self.collector.record_inference(
            db=self.db,
            tenant_id="t1",
            model_id="m1",
            domain="tabular",
            latency_ms=150.0,
            success=True,
            tags={"algorithm": "rf"},
        )
        self.db.add.assert_called_once()
        added = self.db.add.call_args[0][0]
        assert added.tenant_id == "t1"
        assert added.metric_name == "inference_latency_ms"
        assert added.metric_value == 150.0
        assert added.tags == {"algorithm": "rf"}

    def test_record_inference_with_tokens(self):
        self.collector.record_inference(
            db=self.db,
            tenant_id="t1",
            model_id="m1",
            domain="llm",
            latency_ms=500.0,
            tokens_input=150,
            tokens_output=300,
        )
        self.db.add.assert_called_once()
        added = self.db.add.call_args[0][0]
        assert added.domain == "llm"
        assert added.metric_value == 500.0

    def test_record_inference_error(self):
        self.collector.record_inference(
            db=self.db,
            tenant_id="t1",
            model_id="m1",
            domain="tabular",
            latency_ms=200.0,
            success=False,
        )
        self.db.add.assert_called_once()

    def test_record_training(self):
        self.collector.record_training(
            db=self.db,
            tenant_id="t1",
            model_id="m1",
            domain="tabular",
            duration_s=3600.0,
            cpu_hours=1.0,
            tags={"algorithm": "random_forest"},
        )
        self.db.add.assert_called_once()
        added = self.db.add.call_args[0][0]
        assert added.metric_name == "training_duration_s"
        assert added.metric_value == 3600.0

    def test_record_storage(self):
        # No necesita db — solo actualiza gauge
        self.collector.record_storage(
            tenant_id="t1",
            model_id="m1",
            bytes_count=1048576,
        )
        # No debería fallar
        assert True

    def test_record_ws_connection(self):
        self.collector.record_ws_connection(tenant_id="t1", delta=1)
        self.collector.record_ws_connection(tenant_id="t1", delta=-1)

    def test_singleton(self):
        c1 = get_usage_collector()
        c2 = get_usage_collector()
        assert c1 is c2
        assert isinstance(c1, UsageCollector)
