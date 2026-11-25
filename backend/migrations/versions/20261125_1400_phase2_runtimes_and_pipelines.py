"""
Runtime Environments & User-Defined Pipelines con soporte RLS en PostgreSQL.
"""

from datetime import datetime, timezone
from uuid import uuid4

import sqlalchemy as sa
from alembic import op

revision: str = "phase2_runtimes_001"
down_revision: str | None = "phase2_enterprise_001"
branch_labels: str | None = None
depends_on: str | None = None


def _is_postgresql() -> bool:
    return op.get_bind().dialect.name == "postgresql"


def _table_exists(table_name: str) -> bool:
    inspector = sa.inspect(op.get_bind())
    return table_name in inspector.get_table_names()


def upgrade() -> None:
    if not _table_exists("runtime_environments"):
        op.create_table(
            "runtime_environments",
            sa.Column("id", sa.String(), primary_key=True),
            sa.Column("tenant_id", sa.String(), sa.ForeignKey("tenant.id"), nullable=False),
            sa.Column("workspace_id", sa.String(), sa.ForeignKey("workspaces.id"), nullable=True),
            sa.Column("name", sa.String(), nullable=False),
            sa.Column("slug", sa.String(), nullable=False),
            sa.Column("description", sa.String(), nullable=True),
            sa.Column("python_version", sa.String(), nullable=False, server_default="3.10"),
            sa.Column("base_image", sa.String(), nullable=False, server_default="python:3.10-slim"),
            sa.Column("packages", sa.JSON(), nullable=False),
            sa.Column("env_vars", sa.JSON(), nullable=False),
            sa.Column("compute_profile", sa.String(), nullable=False, server_default="cpu-heavy"),
            sa.Column("runner_type", sa.String(), nullable=False, server_default="uv_isolated"),
            sa.Column("is_default", sa.Boolean(), nullable=False, server_default=sa.false()),
            sa.Column("created_by", sa.String(), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.Column("updated_at", sa.DateTime(), nullable=False),
            sa.UniqueConstraint("tenant_id", "slug", name="uq_runtime_env_tenant_slug"),
        )
        op.create_index("ix_runtime_environments_tenant_id", "runtime_environments", ["tenant_id"])

    if not _table_exists("user_pipelines"):
        op.create_table(
            "user_pipelines",
            sa.Column("id", sa.String(), primary_key=True),
            sa.Column("tenant_id", sa.String(), sa.ForeignKey("tenant.id"), nullable=False),
            sa.Column("workspace_id", sa.String(), sa.ForeignKey("workspaces.id"), nullable=True),
            sa.Column("environment_id", sa.String(), sa.ForeignKey("runtime_environments.id"), nullable=True),
            sa.Column("name", sa.String(), nullable=False),
            sa.Column("slug", sa.String(), nullable=False),
            sa.Column("description", sa.String(), nullable=True),
            sa.Column("engine", sa.String(), nullable=False, server_default="declarative"),
            sa.Column("steps", sa.JSON(), nullable=False),
            sa.Column("custom_code", sa.Text(), nullable=True),
            sa.Column("column_renames", sa.JSON(), nullable=False),
            sa.Column("target_column", sa.String(), nullable=True),
            sa.Column("version", sa.Integer(), nullable=False, server_default="1"),
            sa.Column("created_by", sa.String(), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.Column("updated_at", sa.DateTime(), nullable=False),
            sa.UniqueConstraint("tenant_id", "slug", name="uq_user_pipelines_tenant_slug"),
        )
        op.create_index("ix_user_pipelines_tenant_id", "user_pipelines", ["tenant_id"])

    # Seed default runtime environment for existing tenants
    conn = op.get_bind()
    now = datetime.now(timezone.utc)
    tenants = conn.execute(sa.text("SELECT id FROM tenant")).fetchall()
    for (tenant_id,) in tenants:
        existing = conn.execute(
            sa.text("SELECT id FROM runtime_environments WHERE tenant_id = :tid AND slug = 'default-ml-py310'"),
            {"tid": tenant_id},
        ).fetchone()
        if not existing:
            conn.execute(
                sa.text(
                    "INSERT INTO runtime_environments "
                    "(id, tenant_id, name, slug, description, python_version, base_image, packages, env_vars, compute_profile, runner_type, is_default, created_at, updated_at) "
                    "VALUES (:id, :tid, 'Default ML Python 3.10', 'default-ml-py310', "
                    "'Entorno base con pandas, numpy y scikit-learn.', '3.10', 'python:3.10-slim', "
                    "'[\"pandas\", \"numpy\", \"scikit-learn\"]', '{}', 'cpu-heavy', 'uv_isolated', :is_def, :now, :now)"
                ),
                {"id": str(uuid4()), "tid": tenant_id, "is_def": True, "now": now},
            )

    if _is_postgresql():
        for tbl in ["runtime_environments", "user_pipelines"]:
            op.execute(f"ALTER TABLE {tbl} ENABLE ROW LEVEL SECURITY")
            op.execute(f"DROP POLICY IF EXISTS tenant_isolation_{tbl} ON {tbl}")
            op.execute(
                f"CREATE POLICY tenant_isolation_{tbl} ON {tbl} "
                f"USING (current_setting('app.current_tenant_id', true) IS NULL "
                f"OR current_setting('app.current_tenant_id', true) = '' "
                f"OR tenant_id = current_setting('app.current_tenant_id', true))"
            )


def downgrade() -> None:
    for tbl in ["user_pipelines", "runtime_environments"]:
        if _table_exists(tbl):
            op.drop_table(tbl)
