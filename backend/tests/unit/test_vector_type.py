"""
Tests del tipo VectorType: compila a vector(384) en PostgreSQL y JSON en SQLite.
"""

from sqlalchemy import create_engine, inspect
from sqlalchemy.dialects import postgresql
from sqlalchemy.schema import CreateTable

from app.models.base import Base
from app.models.embedding import ModelEmbedding, VectorType

import app.models  # noqa: F401


def test_postgres_ddl_uses_vector():
    sql = str(CreateTable(ModelEmbedding.__table__).compile(dialect=postgresql.dialect()))
    assert "VECTOR(384)" in sql


def test_sqlite_uses_json():
    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=eng)
    insp = inspect(eng)
    cols = {c["name"]: str(c["type"]) for c in insp.get_columns("model_embeddings")}
    assert cols["embedding"] == "JSON"


def test_vector_type_roundtrip_sqlite():
    """Guarda y lee un vector como lista en SQLite."""
    from app.models.embedding import ModelEmbedding

    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=eng)
    from sqlalchemy.orm import sessionmaker

    S = sessionmaker(bind=eng)
    db = S()
    vec = [0.1] * 384
    db.add(
        ModelEmbedding(
            tenant_id="t1",
            model_id="m1",
            embedding=vec,
            metadata_json={"index_type": "model", "model_id": "m1"},
        )
    )
    db.commit()
    row = db.query(ModelEmbedding).first()
    assert row.embedding == vec
    assert row.metadata_json["index_type"] == "model"


def test_vector_type_dimensions():
    assert VectorType(384).dimensions == 384
