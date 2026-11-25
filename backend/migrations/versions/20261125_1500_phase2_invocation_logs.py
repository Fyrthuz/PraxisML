"""
Tabla endpoint_invocation_logs para historial ligero de tráfico por ModelEndpoint
con política PostgreSQL RLS.
"""

import sqlalchemy as sa
from alembic import op

revision: str = "phase2_inv_logs_001"
down_revision: str | None = "phase2_runtimes_001"
branch_labels: str | None = None
depends_on: str | None = None


def _is_postgresql() -> bool:
    return op.get_bind().dialect.name == "postgresql"


def _table_exists(table_name: str) -> bool:
    inspector = sa.inspect(op.get_bind())
    return table_name in inspector.get_table_names()


def upgrade() -> None:
    if not _table_exists("endpoint_invocation_logs"):
        op.create_table(
            "endpoint_invocation_logs",
            sa.Column("id", sa.String(), primary_key=True),
            sa.Column("endpoint_id", sa.String(), sa.ForeignKey("model_endpoints.id"), nullable=False),
            sa.Column("tenant_id", sa.String(), sa.ForeignKey("tenant.id"), nullable=False),
            sa.Column("model_id", sa.String(), nullable=False),
            sa.Column("variant", sa.String(), nullable=False, server_default="primary"),
            sa.Column("runtime", sa.String(), nullable=False, server_default="tabular_generic"),
            sa.Column("status", sa.String(), nullable=False, server_default="ok"),
            sa.Column("latency_ms", sa.Float(), nullable=False, server_default="0"),
            sa.Column("input_count", sa.Integer(), nullable=False, server_default="1"),
            sa.Column("sample_input", sa.JSON(), nullable=True),
            sa.Column("sample_output", sa.JSON(), nullable=True),
            sa.Column("error_message", sa.String(), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=False),
        )
        op.create_index(
            "ix_endpoint_inv_logs_ep_created",
            "endpoint_invocation_logs",
            ["endpoint_id", "created_at"],
        )
        op.create_index(
            "ix_endpoint_inv_logs_tenant_id",
            "endpoint_invocation_logs",
            ["tenant_id"],
        )

    if _is_postgresql():
        tbl = "endpoint_invocation_logs"
        op.execute(f"ALTER TABLE {tbl} ENABLE ROW LEVEL SECURITY")
        op.execute(f"DROP POLICY IF EXISTS tenant_isolation_{tbl} ON {tbl}")
        op.execute(
            f"CREATE POLICY tenant_isolation_{tbl} ON {tbl} "
            f"USING (current_setting('app.current_tenant_id', true) IS NULL "
            f"OR current_setting('app.current_tenant_id', true) = '' "
            f"OR tenant_id = current_setting('app.current_tenant_id', true))"
        )


def downgrade() -> None:
    if _table_exists("endpoint_invocation_logs"):
        op.drop_table("endpoint_invocation_logs")
