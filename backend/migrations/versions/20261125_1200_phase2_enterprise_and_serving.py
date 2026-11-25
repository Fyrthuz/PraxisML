"""
Enterprise Core & Model Serving Endpoints: workspaces, api_keys, audit_log,
model_endpoints y políticas PostgreSQL Row-Level Security (RLS).
"""

from datetime import datetime, timezone
from uuid import uuid4

import sqlalchemy as sa
from alembic import op

revision: str = "phase2_enterprise_001"
down_revision: str | None = "phase1_nlp_001"
branch_labels: str | None = None
depends_on: str | None = None


def _is_postgresql() -> bool:
    return op.get_bind().dialect.name == "postgresql"


def _table_exists(table_name: str) -> bool:
    inspector = sa.inspect(op.get_bind())
    return table_name in inspector.get_table_names()


def upgrade() -> None:
    if not _table_exists("workspaces"):
        op.create_table(
            "workspaces",
            sa.Column("id", sa.String(), primary_key=True),
            sa.Column("tenant_id", sa.String(), sa.ForeignKey("tenant.id"), nullable=False),
            sa.Column("name", sa.String(), nullable=False),
            sa.Column("slug", sa.String(), nullable=False),
            sa.Column("environment", sa.String(), nullable=False, server_default="dev"),
            sa.Column("default_compute_profile", sa.String(), nullable=False, server_default="cpu-light"),
            sa.Column("budget_limit_usd", sa.Float(), nullable=True),
            sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.UniqueConstraint("tenant_id", "slug", name="uq_workspaces_tenant_slug"),
        )
        op.create_index("ix_workspaces_tenant_id", "workspaces", ["tenant_id"])
        op.create_index("ix_workspaces_slug", "workspaces", ["slug"])

    if not _table_exists("api_keys"):
        op.create_table(
            "api_keys",
            sa.Column("id", sa.String(), primary_key=True),
            sa.Column("tenant_id", sa.String(), sa.ForeignKey("tenant.id"), nullable=False),
            sa.Column("workspace_id", sa.String(), sa.ForeignKey("workspaces.id"), nullable=True),
            sa.Column("user_id", sa.String(), sa.ForeignKey("users.id"), nullable=False),
            sa.Column("name", sa.String(), nullable=False),
            sa.Column("key_prefix", sa.String(20), nullable=False),
            sa.Column("key_hash_sha256", sa.String(64), nullable=False, unique=True),
            sa.Column("scopes", sa.JSON(), nullable=False),
            sa.Column("expires_at", sa.DateTime(), nullable=True),
            sa.Column("last_used_at", sa.DateTime(), nullable=True),
            sa.Column("revoked_at", sa.DateTime(), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=False),
        )
        op.create_index("ix_api_keys_tenant_id", "api_keys", ["tenant_id"])
        op.create_index("ix_api_keys_user_id", "api_keys", ["user_id"])
        op.create_index("ix_api_keys_hash", "api_keys", ["key_hash_sha256"], unique=True)

    if not _table_exists("audit_log"):
        op.create_table(
            "audit_log",
            sa.Column("id", sa.String(), primary_key=True),
            sa.Column("tenant_id", sa.String(), sa.ForeignKey("tenant.id"), nullable=False),
            sa.Column("workspace_id", sa.String(), nullable=True),
            sa.Column("actor_id", sa.String(), nullable=True),
            sa.Column("actor_type", sa.String(), nullable=False, server_default="user"),
            sa.Column("action", sa.String(), nullable=False),
            sa.Column("resource_type", sa.String(), nullable=False),
            sa.Column("resource_id", sa.String(), nullable=True),
            sa.Column("details", sa.JSON(), nullable=True),
            sa.Column("ip_address", sa.String(), nullable=True),
            sa.Column("request_id", sa.String(), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=False),
        )
        op.create_index("ix_audit_log_tenant_created", "audit_log", ["tenant_id", "created_at"])
        op.create_index("ix_audit_log_action", "audit_log", ["action"])

    if not _table_exists("model_endpoints"):
        op.create_table(
            "model_endpoints",
            sa.Column("id", sa.String(), primary_key=True),
            sa.Column("tenant_id", sa.String(), sa.ForeignKey("tenant.id"), nullable=False),
            sa.Column("workspace_id", sa.String(), sa.ForeignKey("workspaces.id"), nullable=True),
            sa.Column("name", sa.String(), nullable=False),
            sa.Column("slug", sa.String(), nullable=False, unique=True),
            sa.Column("description", sa.String(), nullable=True),
            sa.Column("model_id", sa.String(), sa.ForeignKey("ml_model.id"), nullable=False),
            sa.Column("secondary_model_id", sa.String(), sa.ForeignKey("ml_model.id"), nullable=True),
            sa.Column("traffic_split_pct", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("preprocessing_pipeline_path", sa.String(), nullable=True),
            sa.Column("status", sa.String(), nullable=False, server_default="active"),
            sa.Column("auth_mode", sa.String(), nullable=False, server_default="api_key"),
            sa.Column("request_count", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("error_count", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("avg_latency_ms", sa.Float(), nullable=False, server_default="0"),
            sa.Column("last_invoked_at", sa.DateTime(), nullable=True),
            sa.Column("created_by", sa.String(), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.Column("updated_at", sa.DateTime(), nullable=False),
        )
        op.create_index("ix_model_endpoints_tenant_id", "model_endpoints", ["tenant_id"])
        op.create_index("ix_model_endpoints_slug", "model_endpoints", ["slug"], unique=True)
        op.create_index("ix_model_endpoints_model_id", "model_endpoints", ["model_id"])

    # Seed default workspace for existing tenants
    conn = op.get_bind()
    now = datetime.now(timezone.utc)
    tenants = conn.execute(sa.text("SELECT id FROM tenant")).fetchall()
    for (tenant_id,) in tenants:
        existing = conn.execute(
            sa.text("SELECT id FROM workspaces WHERE tenant_id = :tid AND slug = 'default'"),
            {"tid": tenant_id},
        ).fetchone()
        if not existing:
            conn.execute(
                sa.text(
                    "INSERT INTO workspaces (id, tenant_id, name, slug, environment, default_compute_profile, is_active, created_at) "
                    "VALUES (:id, :tid, 'Default Workspace', 'default', 'dev', 'cpu-light', :active, :created_at)"
                ),
                {
                    "id": str(uuid4()),
                    "tid": tenant_id,
                    "active": True,
                    "created_at": now,
                },
            )

    # Optional Row-Level Security policies on PostgreSQL
    if _is_postgresql():
        rls_tables = ["workspaces", "api_keys", "audit_log", "model_endpoints"]
        for tbl in rls_tables:
            op.execute(f"ALTER TABLE {tbl} ENABLE ROW LEVEL SECURITY")
            op.execute(f"DROP POLICY IF EXISTS tenant_isolation_{tbl} ON {tbl}")
            op.execute(
                f"CREATE POLICY tenant_isolation_{tbl} ON {tbl} "
                f"USING (current_setting('app.current_tenant_id', true) IS NULL "
                f"OR current_setting('app.current_tenant_id', true) = '' "
                f"OR tenant_id = current_setting('app.current_tenant_id', true))"
            )


def downgrade() -> None:
    for tbl in ["model_endpoints", "audit_log", "api_keys", "workspaces"]:
        if _table_exists(tbl):
            op.drop_table(tbl)
