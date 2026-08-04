"""
Fase 0 — Observabilidad: tablas de métricas, costes, drift, alertas y precios.

Crea 5 tablas nuevas:
  - usage_metric      (métricas atómicas de inferencia/entrenamiento)
  - cost_record       (registros de coste por tenant/recurso)
  - drift_history     (historial de drift por modelo)
  - alert_config      (configuración de alertas por tenant)
  - resource_pricing  (precios dinámicos GPU/CPU/storage/LLM)

Añade 4 columnas a tenant para budget/facturación.

Seed inicial de 16 precios en resource_pricing.
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID

revision: str = "f0_observability_001"
down_revision: str | None = "9b2c3d4e5f6a"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    # ── usage_metric ───────────────────────────────────────────────────────────
    op.create_table(
        "usage_metric",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("tenant_id", sa.String(), nullable=False, index=True),
        sa.Column("model_id", sa.String(), nullable=True),
        sa.Column("domain", sa.String(), nullable=False),
        sa.Column("metric_name", sa.String(), nullable=False),
        sa.Column("metric_value", sa.Float(), nullable=False),
        sa.Column("tags", sa.JSON(), nullable=True),
        sa.Column("recorded_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_usage_tenant_time", "usage_metric", ["tenant_id", "recorded_at"])
    op.create_index(
        "idx_usage_model_metric", "usage_metric", ["model_id", "metric_name", "recorded_at"]
    )

    # ── cost_record ────────────────────────────────────────────────────────────
    op.create_table(
        "cost_record",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("tenant_id", sa.String(), nullable=False),
        sa.Column("model_id", sa.String(), nullable=True),
        sa.Column("cost_type", sa.String(), nullable=False),
        sa.Column("amount", sa.Float(), nullable=False),
        sa.Column("currency", sa.String(), nullable=True, server_default="USD"),
        sa.Column("source", sa.String(), nullable=True),
        sa.Column("period_start", sa.DateTime(timezone=True), nullable=False),
        sa.Column("period_end", sa.DateTime(timezone=True), nullable=False),
        sa.Column("extra_metadata", sa.JSON(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_cost_tenant_period", "cost_record", ["tenant_id", "period_start"])

    # ── drift_history ──────────────────────────────────────────────────────────
    op.create_table(
        "drift_history",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("model_id", sa.String(), nullable=False),
        sa.Column("dataset_id", sa.String(), nullable=True),
        sa.Column("tenant_id", sa.String(), nullable=False),
        sa.Column("drift_type", sa.String(), nullable=False),
        sa.Column("metrics", sa.JSON(), nullable=False),
        sa.Column("threshold_config", sa.JSON(), nullable=True),
        sa.Column("is_anomaly", sa.Boolean(), nullable=True, server_default=sa.text("false")),
        sa.Column("recorded_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_drift_model_time", "drift_history", ["model_id", "recorded_at"])

    # ── alert_config ───────────────────────────────────────────────────────────
    op.create_table(
        "alert_config",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("tenant_id", sa.String(), nullable=False),
        sa.Column("alert_type", sa.String(), nullable=False),
        sa.Column("metric_name", sa.String(), nullable=False),
        sa.Column("threshold", sa.Float(), nullable=False),
        sa.Column("comparison", sa.String(), nullable=False),
        sa.Column("cooldown_min", sa.Integer(), nullable=True, server_default="60"),
        sa.Column("channels", sa.JSON(), nullable=False),
        sa.Column("enabled", sa.Boolean(), nullable=True, server_default=sa.text("true")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )

    # ── resource_pricing ───────────────────────────────────────────────────────
    op.create_table(
        "resource_pricing",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("resource_category", sa.String(), nullable=False),
        sa.Column("resource_id", sa.String(), nullable=False),
        sa.Column("pricing_unit", sa.String(), nullable=False),
        sa.Column("price_per_unit", sa.Float(), nullable=False),
        sa.Column("currency", sa.String(), nullable=True, server_default="USD"),
        sa.Column("extra_metadata", sa.JSON(), nullable=True),
        sa.Column("effective_from", sa.DateTime(timezone=True), nullable=False),
        sa.Column("effective_to", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "idx_resource_pricing_active",
        "resource_pricing",
        ["resource_category", "resource_id", "pricing_unit"],
        unique=True,
        postgresql_where=sa.text("effective_to IS NULL"),
    )

    # ── Extensión de tenant (budget) ──────────────────────────────────────────
    op.add_column("tenant", sa.Column("monthly_budget_usd", sa.Float(), nullable=True))
    op.add_column(
        "tenant",
        sa.Column("budget_alert_pct", sa.Float(), nullable=True, server_default="80.0"),
    )
    op.add_column(
        "tenant",
        sa.Column("billing_cycle", sa.String(), nullable=True, server_default="monthly"),
    )
    op.add_column(
        "tenant",
        sa.Column("stripe_customer_id", sa.String(), nullable=True),
    )

    # ── Seed de precios iniciales ──────────────────────────────────────────────
    from app.utils.uuid import generate_uuid
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    prices = [
        # GPU
        ("compute_gpu", "A100", "per_hour", 3.50),
        ("compute_gpu", "V100", "per_hour", 2.00),
        ("compute_gpu", "T4", "per_hour", 0.80),
        ("compute_gpu", "L4", "per_hour", 1.20),
        # CPU
        ("compute_cpu", "standard", "per_hour", 0.10),
        # Storage
        ("storage", "standard", "per_gb_month", 0.10),
        # LLM Tokens — OpenAI
        ("llm_tokens", "gpt-4", "per_1k_input_tokens", 0.03),
        ("llm_tokens", "gpt-4", "per_1k_output_tokens", 0.06),
        ("llm_tokens", "gpt-4-turbo", "per_1k_input_tokens", 0.01),
        ("llm_tokens", "gpt-4-turbo", "per_1k_output_tokens", 0.03),
        ("llm_tokens", "gpt-3.5", "per_1k_input_tokens", 0.0015),
        ("llm_tokens", "gpt-3.5", "per_1k_output_tokens", 0.002),
        # LLM Tokens — Anthropic
        ("llm_tokens", "claude-3-opus", "per_1k_input_tokens", 0.015),
        ("llm_tokens", "claude-3-opus", "per_1k_output_tokens", 0.075),
        ("llm_tokens", "claude-3-sonnet", "per_1k_input_tokens", 0.003),
        ("llm_tokens", "claude-3-sonnet", "per_1k_output_tokens", 0.015),
    ]

    for category, resource_id, unit, price in prices:
        op.execute(
            f"""
            INSERT INTO resource_pricing (id, resource_category, resource_id, pricing_unit,
                                           price_per_unit, effective_from, created_at)
            VALUES ('{generate_uuid()}', '{category}', '{resource_id}', '{unit}',
                    {price}, '{now.isoformat()}', '{now.isoformat()}')
            """
        )


def downgrade() -> None:
    op.drop_table("resource_pricing")
    op.drop_table("alert_config")
    op.drop_table("drift_history")
    op.drop_table("cost_record")
    op.drop_table("usage_metric")
    op.drop_column("tenant", "monthly_budget_usd")
    op.drop_column("tenant", "budget_alert_pct")
    op.drop_column("tenant", "billing_cycle")
    op.drop_column("tenant", "stripe_customer_id")
