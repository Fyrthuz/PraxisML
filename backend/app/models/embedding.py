"""
Modelo ORM para embeddings semánticos (pgvector) + tipo VectorType dual.

VectorType es un TypeDecorator: en PostgreSQL delega en pgvector.sqlalchemy.Vector
(vector(384) + operador <=>), y en SQLite usa JSON (lista) para poder ejecutar
los tests de integración sin servidor pgvector.
"""

import logging
from datetime import datetime, timezone

from sqlalchemy import JSON, Column, DateTime, String
from sqlalchemy.types import TypeDecorator

from app.core.config import settings
from app.models.base import Base
from app.utils.uuid import generate_uuid

logger = logging.getLogger(__name__)


class VectorType(TypeDecorator):
    """Tipo vectorial pgvector con fallback JSON para SQLite (tests)."""

    impl = JSON
    cache_ok = True

    def __init__(self, dimensions: int = settings.EMBEDDING_DIM):
        self.dimensions = dimensions
        super().__init__()

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            from pgvector.sqlalchemy import Vector

            return dialect.type_descriptor(Vector(self.dimensions))
        return dialect.type_descriptor(JSON())

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        if hasattr(value, "tolist"):
            return value.tolist()
        return value

    def process_result_value(self, value, dialect):
        return value

    class Comparator(TypeDecorator.Comparator):
        """Operador `<=>` (coseno) — solo válido en PostgreSQL."""

        def cosine_distance(self, other):
            return self.op("<=>")(other)

    comparator_factory = Comparator


class ModelEmbedding(Base):
    """Embedding semántico de un modelo o dataset del tenant."""

    __tablename__ = "model_embeddings"

    id = Column(String, primary_key=True, default=generate_uuid, index=True)
    tenant_id = Column(String, nullable=False, index=True)
    model_id = Column(String, nullable=True, index=True)
    dataset_id = Column(String, nullable=True, index=True)
    embedding = Column(VectorType(settings.EMBEDDING_DIM), nullable=False)
    # Columna física "metadata" (metadata es palabra reservada en declarative)
    metadata_json = Column("metadata", JSON, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
