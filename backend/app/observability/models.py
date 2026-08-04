from datetime import datetime, timezone

from sqlalchemy import Boolean, Column, DateTime, Float, Index, Integer, String, JSON, text

from app.models.base import Base
from app.utils.uuid import generate_uuid


class UsageMetric(Base):
    __tablename__ = "usage_metric"

    id = Column(String, primary_key=True, default=generate_uuid)
    tenant_id = Column(String, nullable=False, index=True)
    model_id = Column(String, nullable=True)
    domain = Column(String, nullable=False)
    metric_name = Column(String, nullable=False)
    metric_value = Column(Float, nullable=False)
    tags = Column(JSON, nullable=True)
    recorded_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        Index("idx_usage_tenant_time", "tenant_id", "recorded_at"),
        Index("idx_usage_model_metric", "model_id", "metric_name", "recorded_at"),
    )


class CostRecord(Base):
    __tablename__ = "cost_record"

    id = Column(String, primary_key=True, default=generate_uuid)
    tenant_id = Column(String, nullable=False)
    model_id = Column(String, nullable=True)
    cost_type = Column(String, nullable=False)
    amount = Column(Float, nullable=False)
    currency = Column(String, default="USD")
    source = Column(String, nullable=True)
    period_start = Column(DateTime(timezone=True), nullable=False)
    period_end = Column(DateTime(timezone=True), nullable=False)
    extra_metadata = Column(JSON, nullable=True)

    __table_args__ = (
        Index("idx_cost_tenant_period", "tenant_id", "period_start"),
    )


class DriftHistory(Base):
    __tablename__ = "drift_history"

    id = Column(String, primary_key=True, default=generate_uuid)
    model_id = Column(String, nullable=False)
    dataset_id = Column(String, nullable=True)
    tenant_id = Column(String, nullable=False)
    drift_type = Column(String, nullable=False)
    metrics = Column(JSON, nullable=False)
    threshold_config = Column(JSON, nullable=True)
    is_anomaly = Column(Boolean, default=False)
    recorded_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        Index("idx_drift_model_time", "model_id", "recorded_at"),
    )


class AlertConfig(Base):
    __tablename__ = "alert_config"

    id = Column(String, primary_key=True, default=generate_uuid)
    tenant_id = Column(String, nullable=False)
    alert_type = Column(String, nullable=False)
    metric_name = Column(String, nullable=False)
    threshold = Column(Float, nullable=False)
    comparison = Column(String, nullable=False)
    cooldown_min = Column(Integer, default=60)
    channels = Column(JSON, nullable=False)
    enabled = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class ResourcePricing(Base):
    __tablename__ = "resource_pricing"

    id = Column(String, primary_key=True, default=generate_uuid)
    resource_category = Column(String, nullable=False)
    resource_id = Column(String, nullable=False)
    pricing_unit = Column(String, nullable=False)
    price_per_unit = Column(Float, nullable=False)
    currency = Column(String, default="USD")
    extra_metadata = Column(JSON, nullable=True)
    effective_from = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    effective_to = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        Index(
            "idx_resource_pricing_active",
            "resource_category", "resource_id", "pricing_unit",
            unique=True,
            postgresql_where=text("effective_to IS NULL"),
        ),
    )
