"""Add model registry version and dataset dvc_commit_hash.

Revision ID: a1b2c3d4e5f6
Revises: 093422de8ff9
Create Date: 2026-03-10 12:00:00.000000
"""

from alembic import op
import sqlalchemy as sa


revision: str = "a1b2c3d4e5f6"
down_revision: str | None = "093422de8ff9"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    op.add_column(
        "ml_model",
        sa.Column("version", sa.String(), nullable=False, server_default="1.0.0"),
    )
    op.add_column(
        "ml_model",
        sa.Column("stage", sa.String(), nullable=False, server_default="Staging"),
    )
    op.add_column("ml_model", sa.Column("promoted_at", sa.DateTime(), nullable=True))
    op.add_column("ml_model", sa.Column("promoted_by", sa.String(), nullable=True))
    op.add_column(
        "ml_model", sa.Column("mlflow_registry_name", sa.String(), nullable=True)
    )
    op.add_column("ml_model", sa.Column("mlflow_version", sa.Integer(), nullable=True))

    op.add_column("dataset", sa.Column("dvc_commit_hash", sa.String(), nullable=True))


def downgrade() -> None:
    op.drop_column("dataset", "dvc_commit_hash")
    op.drop_column("ml_model", "mlflow_version")
    op.drop_column("ml_model", "mlflow_registry_name")
    op.drop_column("ml_model", "promoted_by")
    op.drop_column("ml_model", "promoted_at")
    op.drop_column("ml_model", "stage")
    op.drop_column("ml_model", "version")
