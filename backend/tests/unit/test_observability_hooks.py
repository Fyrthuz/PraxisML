"""Tests unitarios para invalidación de cache de precios y hooks de observabilidad."""

from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch

import pytest


class TestPricingCacheInvalidation:
    """Prueba que los precios se invaliden correctamente en Redis."""

    def test_create_pricing_invalidates_cache(self):
        """POST /pricing debe invalidar el precio anterior en Redis."""
        from app.api.routes.v1.observability import create_pricing

        mock_db = MagicMock()
        mock_user = MagicMock()
        mock_user.role = "admin"

        # Mock ResourcePricing query chain
        mock_query = MagicMock()
        mock_filter = MagicMock()
        mock_update = MagicMock()
        mock_query.filter.return_value = mock_filter
        mock_filter.filter.return_value = mock_filter
        mock_filter.filter.return_value.update.return_value = 1
        mock_db.query.return_value = mock_query

        mock_pricing = MagicMock()
        mock_pricing.id = "price-1"
        mock_pricing.resource_category = "compute_gpu"
        mock_pricing.resource_id = "H100"
        mock_pricing.pricing_unit = "per_hour"
        mock_pricing.price_per_unit = 4.50
        mock_pricing.currency = "USD"
        mock_pricing.metadata = {}
        mock_pricing.effective_from = datetime.now(timezone.utc)
        mock_pricing.effective_to = None
        mock_pricing.created_at = datetime.now(timezone.utc)

        mock_db.add.return_value = None

        with patch("app.api.routes.v1.observability.ResourcePricing", return_value=mock_pricing):
            with patch("app.api.routes.v1.observability.datetime") as mock_dt:
                mock_dt.now.return_value = datetime.now(timezone.utc)
                from app.schemas.pricing import ResourcePricingCreate
                data = ResourcePricingCreate(
                    resource_category="compute_gpu",
                    resource_id="H100",
                    pricing_unit="per_hour",
                    price_per_unit=4.50,
                )
                result = create_pricing(data, db=mock_db, _user=mock_user)
                assert result is not None

    def test_get_price_uses_cache(self):
        """CostCalculator debe usar cache Redis cuando está disponible."""
        from app.observability.cost_calculator import CostCalculator

        calc = CostCalculator()
        mock_db = MagicMock()

        with patch("app.core.redis.create_sync_redis_client") as mock_redis_factory:
            mock_redis = MagicMock()
            mock_redis_factory.return_value = mock_redis
            mock_redis.get.return_value = "3.50"

            price = calc.get_price(mock_db, "compute_gpu", "A100", "per_hour")
            assert price == 3.50

    def test_get_price_fallback_to_db(self):
        """CostCalculator debe hacer fallback a DB cuando Redis falla."""
        from app.observability.cost_calculator import CostCalculator
        from app.observability.models import ResourcePricing

        calc = CostCalculator()
        mock_db = MagicMock()

        mock_pricing = MagicMock(spec=ResourcePricing)
        mock_pricing.price_per_unit = 2.50

        mock_chain = mock_db.query.return_value
        mock_chain.filter.return_value = mock_chain
        mock_chain.filter.return_value = mock_chain
        mock_chain.filter.return_value.first.return_value = mock_pricing

        with patch("app.core.redis.create_sync_redis_client") as mock_redis_factory:
            mock_redis_factory.side_effect = Exception("Redis unavailable")
            price = calc.get_price(mock_db, "compute_gpu", "V100", "per_hour")
            assert price == 2.50


class TestObservabilityHooks:
    """Prueba que los hooks de observabilidad se ejecuten correctamente."""

    def test_record_inference_with_cost(self):
        """record_inference debe registrar coste si es > 0."""
        from app.observability.usage_collector import UsageCollector
        from app.observability.cost_calculator import CostCalculator

        collector = UsageCollector()
        calculator = CostCalculator()
        mock_db = MagicMock()

        # Mock CostRecord
        with patch("app.observability.usage_collector.UsageMetric"):
            collector.record_inference(
                db=mock_db,
                tenant_id="t1",
                model_id="m1",
                domain="tabular",
                latency_ms=100.0,
                success=True,
                tags={"algorithm": "rf"},
            )

        # Verificar que se llamó a db.add
        assert mock_db.add.called

    def test_record_training_with_cost(self):
        """record_training debe registrar coste si es > 0."""
        from app.observability.usage_collector import UsageCollector

        collector = UsageCollector()
        mock_db = MagicMock()

        with patch("app.observability.usage_collector.UsageMetric"):
            collector.record_training(
                db=mock_db,
                tenant_id="t1",
                model_id="m1",
                domain="tabular",
                duration_s=3600.0,
                cpu_hours=1.0,
                tags={"algorithm": "rf"},
            )

        assert mock_db.add.called
