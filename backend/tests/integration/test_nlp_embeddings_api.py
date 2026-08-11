"""
Tests de integración para la API de embeddings y búsqueda semántica.
SQLite + override de get_db. Modelo de embeddings mockeado.
"""


import numpy as np
import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.main import create_app
from app.database import get_db
from app.models.base import Base
from app.models.user import User, UserRole
from app.models.tenant import Tenant
from app.core.security import create_access_token, get_password_hash
from app.models.embedding import ModelEmbedding

import app.models.embedding  # noqa: F401
import app.models.assistant  # noqa: F401

_fastapi_app = create_app()
SQLITE_URL = "sqlite:///./test_nlp_embeddings_api.db"


class FakeST:
    def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True):
        texts = [texts] if isinstance(texts, str) else list(texts)
        out = np.zeros((len(texts), 384), dtype=np.float32)
        for i, t in enumerate(texts):
            rng = np.random.RandomState(hash(t) % 2**32)
            v = rng.randn(384).astype(np.float32)
            out[i] = v / (np.linalg.norm(v) + 1e-9)
        return out


@pytest.fixture(scope="module")
def engine():
    eng = create_engine(SQLITE_URL, connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=eng)
    yield eng
    Base.metadata.drop_all(bind=eng)


@pytest.fixture()
def db_session(engine):
    Session = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = Session()
    try:
        yield session
    finally:
        session.rollback()
        session.close()


@pytest.fixture(autouse=True)
def _clean_db(engine):
    with engine.connect() as conn:
        for table in reversed(Base.metadata.sorted_tables):
            conn.execute(table.delete())
        conn.commit()
    yield


@pytest.fixture()
def override_db(db_session):
    _fastapi_app.dependency_overrides[get_db] = lambda: db_session
    yield
    _fastapi_app.dependency_overrides.clear()


@pytest_asyncio.fixture()
async def client(override_db):
    async with AsyncClient(
        transport=ASGITransport(app=_fastapi_app), base_url="http://test"
    ) as ac:
        yield ac


@pytest.fixture(autouse=True)
def mock_embedding_model(monkeypatch):
    monkeypatch.setattr(
        "sentence_transformers.SentenceTransformer", lambda *a, **k: FakeST()
    )


def _create_tenant(db_session, name="Embed Tenant"):
    tenant = Tenant(name=name)
    db_session.add(tenant)
    db_session.commit()
    db_session.refresh(tenant)
    return tenant


def _create_user(db_session, tenant, role):
    import uuid

    email = f"em_{role}_{uuid.uuid4().hex[:8]}@test.com"
    user = User(
        email=email,
        hashed_password=get_password_hash("pass"),
        full_name="Test",
        tenant_id=tenant.id,
        role=role,
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user, create_access_token(subject=user.id)


def _seed_embedding(db_session, tenant, model_id, text, name):
    svc_emb = FakeST().encode([text])[0].tolist()
    db_session.add(ModelEmbedding(
        tenant_id=tenant.id,
        model_id=model_id,
        embedding=svc_emb,
        metadata_json={
            "index_type": "model",
            "model_id": model_id,
            "model_name": name,
            "tenant_id": tenant.id,
        },
    ))
    db_session.commit()


async def test_nlp_embed(client, db_session):
    tenant = _create_tenant(db_session)
    _, token = _create_user(db_session, tenant, UserRole.VIEWER)
    response = await client.post(
        "/api/v1/nlp/embed",
        json={"texts": ["hola", "mundo"]},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["dim"] == 384
    assert len(body["embeddings"]) == 2
    assert len(body["embeddings"][0]) == 384


async def test_nlp_search_returns_tenant_results(client, db_session):
    tenant = _create_tenant(db_session)
    _, token = _create_user(db_session, tenant, UserRole.VIEWER)
    _seed_embedding(db_session, tenant, "m1", "modelo de sentimiento sobre reseñas", "Sentiment")

    other = _create_tenant(db_session, "Other")
    _seed_embedding(db_session, other, "m9", "modelo de sentimiento sobre reseñas", "Sentiment2")

    response = await client.get(
        "/api/v1/nlp/search?q=modelo de sentimiento sobre reseñas&index=models",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    body = response.json()
    assert len(body["results"]) == 1
    assert body["results"][0]["model_id"] == "m1"
    assert body["results"][0]["score"] > 0.5


async def test_nlp_search_invalid_index(client, db_session):
    tenant = _create_tenant(db_session)
    _, token = _create_user(db_session, tenant, UserRole.VIEWER)
    response = await client.get(
        "/api/v1/nlp/search?q=hola&index=bananas",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 400
