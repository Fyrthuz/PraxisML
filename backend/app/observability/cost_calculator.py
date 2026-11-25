"""
CostCalculator — Motor de estimación de costes con precios dinámicos desde DB + Redis cache.
"""

import logging
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.observability.models import CostRecord, ResourcePricing

logger = logging.getLogger(__name__)

REDIS_PRICING_TTL = 300


class CostCalculator:
    def get_price(
        self, db: Session, category: str, resource_id: str, pricing_unit: str
    ) -> Optional[float]:
        cache_key = f"pricing:{category}:{resource_id}:{pricing_unit}"

        try:
            from app.core.redis import create_sync_redis_client

            redis = create_sync_redis_client()
            if redis is not None:
                cached = redis.get(cache_key)
                if cached:
                    return float(cached)
        except Exception:
            pass

        pricing = (
            db.query(ResourcePricing)
            .filter(
                ResourcePricing.resource_category == category,
                ResourcePricing.resource_id == resource_id,
                ResourcePricing.pricing_unit == pricing_unit,
                ResourcePricing.effective_to.is_(None),
            )
            .first()
        )
        if pricing is None:
            logger.warning("Precio no encontrado: %s/%s/%s", category, resource_id, pricing_unit)
            return None

        try:
            from app.core.redis import create_sync_redis_client

            redis = create_sync_redis_client()
            if redis is not None:
                redis.setex(cache_key, REDIS_PRICING_TTL, str(pricing.price_per_unit))
        except Exception:
            pass

        return pricing.price_per_unit

    def estimate_inference_cost(
        self,
        db: Session,
        domain: str,
        latency_ms: float,
        tokens: Optional[dict] = None,
        gpu_type: Optional[str] = None,
    ) -> float:
        total = 0.0

        if domain == "llm" and tokens:
            input_price = self.get_price(db, "llm_tokens", tokens.get("model"), "per_1k_input_tokens")
            output_price = self.get_price(db, "llm_tokens", tokens.get("model"), "per_1k_output_tokens")
            if input_price:
                total += (tokens.get("input", 0) / 1000) * input_price
            if output_price:
                total += (tokens.get("output", 0) / 1000) * output_price
        elif domain in ("tabular", "nlp", "cv", "timeseries"):
            hours = (latency_ms / 1000) / 3600
            if gpu_type:
                gpu_price = self.get_price(db, "compute_gpu", gpu_type, "per_hour")
                if gpu_price:
                    total += hours * gpu_price
            else:
                cpu_price = self.get_price(db, "compute_cpu", "standard", "per_hour")
                if cpu_price:
                    total += hours * cpu_price

        return total

    def estimate_training_cost(
        self,
        db: Session,
        gpu_hours: float = 0,
        gpu_type: Optional[str] = None,
        cpu_hours: float = 0,
    ) -> float:
        total = 0.0
        if gpu_hours > 0 and gpu_type:
            gpu_price = self.get_price(db, "compute_gpu", gpu_type, "per_hour")
            if gpu_price:
                total += gpu_hours * gpu_price
        if cpu_hours > 0:
            cpu_price = self.get_price(db, "compute_cpu", "standard", "per_hour")
            if cpu_price:
                total += cpu_hours * cpu_price
        return total

    def record_cost(
        self,
        db: Session,
        tenant_id: str,
        cost_type: str,
        amount: float,
        source: str,
        period_start: datetime,
        period_end: datetime,
        model_id: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> CostRecord:
        record = CostRecord(
            tenant_id=tenant_id,
            model_id=model_id,
            cost_type=cost_type,
            amount=amount,
            source=source,
            period_start=period_start,
            period_end=period_end,
            extra_metadata=metadata or {},
        )
        db.add(record)

        from app.observability.metrics import inference_cost_usd, training_cost_usd

        if cost_type in ("inference", "serving", "api_tokens"):
            inference_cost_usd.labels(
                tenant_id=tenant_id, model_id=model_id or "unknown", cost_type=cost_type
            ).inc(amount)
        elif cost_type in ("training", "compute"):
            training_cost_usd.labels(
                tenant_id=tenant_id,
                domain=(metadata or {}).get("domain", "unknown"),
                algorithm=(metadata or {}).get("algorithm", "unknown"),
            ).inc(amount)

        self._update_budget_gauge(db, tenant_id)
        return record

    def get_monthly_cost(self, db: Session, tenant_id: str) -> float:
        now = datetime.now(timezone.utc)
        month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

        result = (
            db.query(func.coalesce(func.sum(CostRecord.amount), 0))
            .filter(
                CostRecord.tenant_id == tenant_id,
                CostRecord.period_start >= month_start,
            )
            .scalar()
        )
        return float(result)

    def budget_remaining(self, db: Session, tenant_id: str) -> Optional[float]:
        from app.models.tenant import Tenant

        tenant = db.query(Tenant).filter(Tenant.id == tenant_id).first()
        if not tenant or tenant.monthly_budget_usd is None:
            return None
        spent = self.get_monthly_cost(db, tenant_id)
        return tenant.monthly_budget_usd - spent

    def _update_budget_gauge(self, db: Session, tenant_id: str) -> None:
        remaining = self.budget_remaining(db, tenant_id)
        if remaining is not None:
            from app.observability.metrics import cost_budget_remaining
            cost_budget_remaining.labels(tenant_id=tenant_id).set(max(remaining, 0))


_cost_calculator: CostCalculator | None = None


def get_cost_calculator() -> CostCalculator:
    global _cost_calculator
    if _cost_calculator is None:
        _cost_calculator = CostCalculator()
    return _cost_calculator
