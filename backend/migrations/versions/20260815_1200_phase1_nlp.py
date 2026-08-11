"""
Fase 1 — NLP Engine: embeddings semánticos (pgvector), conversaciones del asistente y profiler.

Cambios:
  - CREATE EXTENSION IF NOT EXISTS vector (solo PostgreSQL)
  - Tabla model_embeddings (id, tenant_id, model_id, dataset_id, embedding vector(384), metadata jsonb)
    + índice IVFFlat (lists=100) sobre embedding (solo PostgreSQL)
  - Tabla assistant_conversations (historial del chatbot RAG)
  - Columna dataset.column_types_analysis JSON (salida del NLP Profiler)
  - Seed de precios vLLM en resource_pricing (llm_tokens, precio 0 self-hosted)
"""

from datetime import datetime, timezone

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect

revision: str = "phase1_nlp_001"
down_revision: str | None = "f0_observability_001"
branch_labels: str | None = None
depends_on: str | None = None


def _is_postgresql() -> bool:
    bind = op.get_bind()
    return bind.dialect.name == "postgresql"


def upgrade() -> None:
    # ── Extensión pgvector (solo PostgreSQL) ──────────────────────────────────
    if _is_postgresql():
        op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # ── model_embeddings ──────────────────────────────────────────────────────
    embedding_type = sa.Text()
    if _is_postgresql():
        embedding_type = sa.Text()  # vector(N) no existe como tipo nativo SA; se usa raw SQL

    op.create_table(
        "model_embeddings",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("tenant_id", sa.String(), nullable=False, index=True),
        sa.Column("model_id", sa.String(), nullable=True, index=True),
        sa.Column("dataset_id", sa.String(), nullable=True, index=True),
        sa.Column("embedding", embedding_type, nullable=False),
        sa.Column("metadata", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_model_embeddings_tenant", "model_embeddings", ["tenant_id"])

    if _is_postgresql():
        # La columna embedding ya se creó como TEXT; la convertimos a vector(384).
        op.execute("ALTER TABLE model_embeddings ALTER COLUMN embedding TYPE vector(384) USING embedding::vector")
        op.execute(
            "CREATE INDEX idx_model_embeddings_ivfflat ON model_embeddings "
            "USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)"
        )

    # ── assistant_conversations ───────────────────────────────────────────────
    op.create_table(
        "assistant_conversations",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("tenant_id", sa.String(), nullable=False, index=True),
        sa.Column("user_id", sa.String(), nullable=True),
        sa.Column("messages", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.Column("updated_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "idx_assistant_conversations_tenant", "assistant_conversations", ["tenant_id"]
    )

    # ── dataset.column_types_analysis ─────────────────────────────────────────
    bind = op.get_bind()
    inspector = inspect(bind)
    if "dataset" in inspector.get_table_names():
        cols = [c["name"] for c in inspector.get_columns("dataset")]
        if "column_types_analysis" not in cols:
            op.add_column(
                "dataset", sa.Column("column_types_analysis", sa.JSON(), nullable=True)
            )

    # ── Seed de precios vLLM (llm_tokens, self-hosted → 0.0) ──────────────────
    from app.utils.uuid import generate_uuid

    now = datetime.now(timezone.utc)
    prices = [
        ("llm_tokens", "Qwen/Qwen2.5-7B-Instruct", "per_1k_input_tokens", 0.0),
        ("llm_tokens", "Qwen/Qwen2.5-7B-Instruct", "per_1k_output_tokens", 0.0),
        ("llm_tokens", "Qwen/Qwen2.5-3B-Instruct", "per_1k_input_tokens", 0.0),
        ("llm_tokens", "Qwen/Qwen2.5-3B-Instruct", "per_1k_output_tokens", 0.0),
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
    if _is_postgresql():
        op.execute("DROP INDEX IF EXISTS idx_model_embeddings_ivfflat")
    op.drop_index("idx_model_embeddings_tenant", table_name="model_embeddings")
    op.drop_table("model_embeddings")
    op.drop_index(
        "idx_assistant_conversations_tenant", table_name="assistant_conversations"
    )
    op.drop_table("assistant_conversations")

    bind = op.get_bind()
    inspector = inspect(bind)
    if "dataset" in inspector.get_table_names():
        cols = [c["name"] for c in inspector.get_columns("dataset")]
        if "column_types_analysis" in cols:
            op.drop_column("dataset", "column_types_analysis")
