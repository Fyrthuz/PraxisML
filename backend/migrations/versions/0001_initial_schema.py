"""
Migración inicial — Estado completo del schema de PraxisML.

Crea todas las tablas desde cero. Si la BD ya tiene las tablas,
esta migración falla con un error "ya existe" — en ese caso usa:

    alembic stamp head      # marca como aplicada sin ejecutarla

Incluye:
  - tenant
  - users  (con columna role para RBAC)
  - dataset
  - ml_model
  - prediction
"""

from alembic import op
import sqlalchemy as sa

revision: str = "0001_initial_schema"
down_revision: str | None = None
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    # ── tenant ────────────────────────────────────────────────────────────────
    op.create_table(
        "tenant",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_tenant_id"), "tenant", ["id"], unique=False)

    # ── users ─────────────────────────────────────────────────────────────────
    op.create_table(
        "users",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("email", sa.String(), nullable=False),
        sa.Column("hashed_password", sa.String(), nullable=True),
        sa.Column("full_name", sa.String(), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=True),
        sa.Column(
            "role",
            sa.String(),
            nullable=False,
            server_default="viewer",
            comment="RBAC role: admin | editor | viewer",
        ),
        sa.Column("tenant_id", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["tenant_id"], ["tenant.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_users_id"), "users", ["id"], unique=False)
    op.create_index(op.f("ix_users_email"), "users", ["email"], unique=True)
    op.create_index(op.f("ix_users_tenant_id"), "users", ["tenant_id"], unique=False)

    # ── dataset ───────────────────────────────────────────────────────────────
    op.create_table(
        "dataset",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("description", sa.String(), nullable=True),
        sa.Column("file_path", sa.String(), nullable=True),
        sa.Column("file_type", sa.String(), nullable=True),
        sa.Column("num_rows", sa.Integer(), nullable=True),
        sa.Column("num_columns", sa.Integer(), nullable=True),
        sa.Column("column_names", sa.JSON(), nullable=True),
        sa.Column("version", sa.Integer(), nullable=True, server_default="1"),
        sa.Column("mlflow_artifact_uri", sa.String(), nullable=True),
        sa.Column("pipeline_path", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.Column("tenant_id", sa.String(), nullable=False),
        sa.ForeignKeyConstraint(["tenant_id"], ["tenant.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_dataset_id"), "dataset", ["id"], unique=False)
    op.create_index(op.f("ix_dataset_tenant_id"), "dataset", ["tenant_id"], unique=False)

    # ── ml_model ──────────────────────────────────────────────────────────────
    op.create_table(
        "ml_model",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("description", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.Column("mlflow_run_id", sa.String(), nullable=False),
        sa.Column("metrics_metadata", sa.JSON(), nullable=True),
        sa.Column("preprocessing_pipeline_path", sa.String(), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=True),
        sa.Column("is_public", sa.Boolean(), nullable=True),
        sa.Column("is_torchscript", sa.Boolean(), nullable=True),
        sa.Column("torchscript_path", sa.String(), nullable=True),
        sa.Column("tenant_id", sa.String(), nullable=False),
        sa.ForeignKeyConstraint(["tenant_id"], ["tenant.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_ml_model_id"), "ml_model", ["id"], unique=False)
    op.create_index(op.f("ix_ml_model_mlflow_run_id"), "ml_model", ["mlflow_run_id"], unique=True)
    op.create_index(op.f("ix_ml_model_tenant_id"), "ml_model", ["tenant_id"], unique=False)

    # ── prediction ────────────────────────────────────────────────────────────
    op.create_table(
        "prediction",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("task_id", sa.String(), nullable=True),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("method", sa.String(), nullable=False),
        sa.Column("result_path", sa.String(), nullable=True),
        sa.Column("uncertainty_path", sa.String(), nullable=True),
        sa.Column("input_image_path", sa.String(), nullable=True),
        sa.Column("mlflow_inference_run_id", sa.String(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
        sa.Column("dataset_id", sa.String(), nullable=True),
        sa.Column("model_id", sa.String(), nullable=False),
        sa.Column("tenant_id", sa.String(), nullable=False),
        sa.ForeignKeyConstraint(["dataset_id"], ["dataset.id"]),
        sa.ForeignKeyConstraint(["model_id"], ["ml_model.id"]),
        sa.ForeignKeyConstraint(["tenant_id"], ["tenant.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_prediction_id"), "prediction", ["id"], unique=False)
    op.create_index(op.f("ix_prediction_task_id"), "prediction", ["task_id"], unique=True)
    op.create_index(op.f("ix_prediction_mlflow_inference_run_id"), "prediction", ["mlflow_inference_run_id"], unique=False)
    op.create_index(op.f("ix_prediction_dataset_id"), "prediction", ["dataset_id"], unique=False)
    op.create_index(op.f("ix_prediction_model_id"), "prediction", ["model_id"], unique=False)
    op.create_index(op.f("ix_prediction_tenant_id"), "prediction", ["tenant_id"], unique=False)


def downgrade() -> None:
    op.drop_table("prediction")
    op.drop_table("ml_model")
    op.drop_table("dataset")
    op.drop_table("users")
    op.drop_table("tenant")
