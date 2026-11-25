"""Tests unitarios para CostCalculator."""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from app.observability.cost_calculator import CostCalculator, get_cost_calculator


class TestCostCalculator:
    def setup_method(self):
        self.calculator = CostCalculator()
        self.db = MagicMock()

    def test_get_price_from_db(self):
        """Cuando no hay cache Redis, consulta DB."""
        from app.observability.models import ResourcePricing

        mock_pricing = MagicMock(spec=ResourcePricing)
        mock_pricing.price_per_unit = 3.50

        query_mock = MagicMock()
        query_mock.filter.return_value.first.return_value = mock_pricing
        self.db.query.return_value = query_mock

        with patch("app.core.redis.create_sync_redis_client", return_value=None):
            price = self.calculator.get_price(self.db, "compute_gpu", "A100", "per_hour")
        assert price == 3.50
        self.db.query.assert_called_once_with(ResourcePricing)

    def test_get_price_not_found(self):
        query_mock = MagicMock()
        query_mock.filter.return_value.first.return_value = None
        self.db.query.return_value = query_mock

        with patch("app.core.redis.create_sync_redis_client", return_value=None):
            price = self.calculator.get_price(self.db, "compute_gpu", "H100", "per_hour")
        assert price is None

    def test_estimate_inference_cost_tabular(self):
        with patch.object(self.calculator, 'get_price', return_value=0.10):
            cost = self.calculator.estimate_inference_cost(
                self.db, domain="tabular", latency_ms=3600000,  # 1 hora
            )
            # 1 hora * $0.10/hora
            assert cost == pytest.approx(0.10, rel=0.01)

    def test_estimate_inference_cost_llm(self):
        def mock_get_price(db, category, resource_id, unit):
            prices = {
                ("llm_tokens", "gpt-4", "per_1k_input_tokens"): 0.03,
                ("llm_tokens", "gpt-4", "per_1k_output_tokens"): 0.06,
            }
            return prices.get((category, resource_id, unit))

        with patch.object(self.calculator, 'get_price', side_effect=mock_get_price):
            cost = self.calculator.estimate_inference_cost(
                self.db, domain="llm",
                latency_ms=500,
                tokens={"model": "gpt-4", "input": 1000, "output": 500},
            )
            # (1000/1000)*0.03 + (500/1000)*0.06 = 0.03 + 0.03 = 0.06
            assert cost == pytest.approx(0.06, rel=0.01)

    def test_estimate_inference_cost_llm_no_tokens(self):
        cost = self.calculator.estimate_inference_cost(
            self.db, domain="llm", latency_ms=500,
        )
        assert cost == 0.0

    def test_estimate_training_cost_gpu(self):
        with patch.object(self.calculator, 'get_price', return_value=3.50):
            cost = self.calculator.estimate_training_cost(
                self.db, gpu_hours=2.0, gpu_type="A100",
            )
            assert cost == pytest.approx(7.0, rel=0.01)

    def test_estimate_training_cost_mixed(self):
        prices = {"compute_gpu": 2.0, "compute_cpu": 0.10}

        def mock_get_price(db, category, resource_id, unit):
            return prices.get(category)

        with patch.object(self.calculator, 'get_price', side_effect=mock_get_price):
            cost = self.calculator.estimate_training_cost(
                self.db, gpu_hours=1.0, gpu_type="V100", cpu_hours=3.0,
            )
            # 1*2.0 + 3*0.10 = 2.0 + 0.30 = 2.30
            assert cost == pytest.approx(2.30, rel=0.01)

    def test_record_cost(self):
        with patch.object(self.calculator, '_update_budget_gauge'):
            result = self.calculator.record_cost(
                db=self.db,
                tenant_id="t1",
                cost_type="inference",
                amount=0.05,
                source="pred-1",
                period_start=datetime.now(timezone.utc),
                period_end=datetime.now(timezone.utc),
                model_id="m1",
            )
            assert result is not None
            self.db.add.assert_called_once()
            added = self.db.add.call_args[0][0]
            assert added.tenant_id == "t1"
            assert added.cost_type == "inference"
            assert added.amount == 0.05

    def test_get_monthly_cost(self):
        self.db.query.return_value.filter.return_value.scalar.return_value = 50.0
        cost = self.calculator.get_monthly_cost(self.db, "t1")
        assert cost == 50.0

    def test_singleton(self):
        c1 = get_cost_calculator()
        c2 = get_cost_calculator()
        assert c1 is c2
        assert isinstance(c1, CostCalculator)
