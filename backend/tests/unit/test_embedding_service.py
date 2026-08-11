"""
Tests del EmbeddingService (modelo de embeddings mockeado).
"""

import numpy as np
import pytest

from app.services.embedding_service import EmbeddingService


class FakeSentenceTransformer:
    def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True):
        texts = [texts] if isinstance(texts, str) else list(texts)
        out = np.zeros((len(texts), 384), dtype=np.float32)
        for i, t in enumerate(texts):
            rng = np.random.RandomState(hash(t) % 2**32)
            v = rng.randn(384).astype(np.float32)
            out[i] = v / (np.linalg.norm(v) + 1e-9)
        return out


@pytest.fixture
def service(monkeypatch):
    svc = EmbeddingService(use_cache=False)
    monkeypatch.setattr(
        "sentence_transformers.SentenceTransformer", lambda *a, **k: FakeSentenceTransformer()
    )
    return svc


def test_embed_texts_shape(service):
    emb = service.embed_texts(["hola mundo", "otro texto"])
    assert emb.shape == (2, 384)


def test_embed_text_single(service):
    vec = service.embed_text("hola")
    assert len(vec) == 384


def test_embed_empty(service):
    emb = service.embed_texts([])
    assert emb.shape == (0, 384)


def test_index_model_upsert(service):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from app.models.base import Base
    from app.models.embedding import ModelEmbedding
    from types import SimpleNamespace

    import app.models  # noqa: F401

    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=eng)
    db = sessionmaker(bind=eng)()

    ml_model = SimpleNamespace(
        id="m1", name="DistilBERT Reviews", description="sentiment",
        tenant_id="t1", stage="Staging", version="1.0.0",
        mlflow_run_id="r1", mlflow_registry_name=None, mlflow_version=None,
        metrics_metadata={
            "domain": "nlp", "algorithm": "distilbert",
            "task_type": "sentiment_analysis", "metrics": {"test_f1": 0.9},
            "dataset_id": "d1", "dataset_name": "reviews.csv",
            "text_column": "text", "label_column": "label",
        },
    )

    service.index_model(db, ml_model)
    db.commit()
    assert db.query(ModelEmbedding).count() == 1

    # Upsert: indexar de nuevo no crea duplicados
    service.index_model(db, ml_model)
    db.commit()
    assert db.query(ModelEmbedding).count() == 1


def test_search_sqlite_isolates_tenant(service):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from app.models.base import Base
    from app.models.embedding import ModelEmbedding

    import app.models  # noqa: F401

    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=eng)
    db = sessionmaker(bind=eng)()

    vec_a = service.embed_texts(["modelo sobre reseñas"])[0].tolist()
    vec_b = service.embed_texts(["modelo sobre finanzas"])[0].tolist()

    db.add(ModelEmbedding(
        tenant_id="t1", model_id="m1", embedding=vec_a,
        metadata_json={"index_type": "model", "model_id": "m1", "model_name": "A"},
    ))
    db.add(ModelEmbedding(
        tenant_id="t2", model_id="m2", embedding=vec_b,
        metadata_json={"index_type": "model", "model_id": "m2", "model_name": "B"},
    ))
    db.commit()

    results = service.search(db, "t1", "modelo sobre reseñas", top_k=5, index="models")
    assert len(results) == 1
    assert results[0]["model_id"] == "m1"
    assert results[0]["score"] > 0.5


def test_search_filters_by_index(service):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from app.models.base import Base
    from app.models.embedding import ModelEmbedding

    import app.models  # noqa: F401

    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=eng)
    db = sessionmaker(bind=eng)()

    vec = service.embed_texts(["texto común"])[0].tolist()
    db.add(ModelEmbedding(
        tenant_id="t1", model_id="m1", embedding=vec,
        metadata_json={"index_type": "model", "model_name": "M"},
    ))
    db.add(ModelEmbedding(
        tenant_id="t1", dataset_id="d1", embedding=vec,
        metadata_json={"index_type": "dataset", "name": "D"},
    ))
    db.commit()

    models = service.search(db, "t1", "texto común", index="models")
    datasets = service.search(db, "t1", "texto común", index="datasets")
    assert len(models) == 1 and models[0]["model_id"] == "m1"
    assert len(datasets) == 1 and datasets[0]["dataset_id"] == "d1"
