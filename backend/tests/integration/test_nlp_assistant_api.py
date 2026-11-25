"""
Tests de integración para la API del asistente (LLM + RAG).
SQLite + override de get_db. Gateway y RAG mockeados.
"""

import json as jsonlib
from unittest.mock import AsyncMock, MagicMock, patch

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

import app.models.embedding  # noqa: F401
import app.models.assistant  # noqa: F401

_fastapi_app = create_app()
SQLITE_URL = "sqlite:///./test_nlp_assistant_api.db"


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


def _create_tenant(db_session, name="Assist Tenant"):
    tenant = Tenant(name=name)
    db_session.add(tenant)
    db_session.commit()
    db_session.refresh(tenant)
    return tenant


def _create_user(db_session, tenant, role):
    import uuid

    email = f"as_{role}_{uuid.uuid4().hex[:8]}@test.com"
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


def _mock_gateway(answer="Tu modelo de sentimiento tiene F1 0.90."):
    gateway = MagicMock()
    gateway.complete = AsyncMock(
        return_value={
            "content": answer,
            "usage": {"input": 100, "output": 50},
            "latency_ms": 300.0,
        }
    )

    async def _stream_gen(*args, **kwargs):
        for w in answer.split():
            yield {"delta": w + " "}
        yield {"usage": {"input": 100, "output": 50}}

    gateway.stream = _stream_gen
    return gateway


def _mock_rag(sources=None):
    rag = MagicMock()
    rag.SYSTEM_PROMPT = "System prompt test"
    rag.retrieve_context.return_value = {
        "context": '{"modelos":[]}',
        "sources": sources or [{"model_id": "m1", "name": "Sentiment Model", "type": "model"}],
    }
    return rag


async def test_assistant_chat_json(client, db_session):
    tenant = _create_tenant(db_session)
    user, token = _create_user(db_session, tenant, UserRole.EDITOR)

    gateway = _mock_gateway()
    rag = _mock_rag()
    with patch("app.domains.nlp.llm_gateway.get_llm_gateway", return_value=gateway), patch(
        "app.domains.nlp.rag_service.get_rag_service", return_value=rag
    ):
        response = await client.post(
            "/api/v1/nlp/assistant/chat",
            json={"message": "¿Cómo está mi modelo?"},
            headers={"Authorization": f"Bearer {token}"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["conversation_id"]
    assert "F1 0.90" in body["answer"]
    assert body["sources"][0]["model_id"] == "m1"
    assert body["tokens"]["input"] == 100


async def test_assistant_chat_stream(client, db_session):
    tenant = _create_tenant(db_session)
    _, token = _create_user(db_session, tenant, UserRole.EDITOR)

    gateway = _mock_gateway("Hola desde streaming")
    rag = _mock_rag()
    with patch("app.domains.nlp.llm_gateway.get_llm_gateway", return_value=gateway), patch(
        "app.domains.nlp.rag_service.get_rag_service", return_value=rag
    ):
        async with client.stream(
            "POST",
            "/api/v1/nlp/assistant/chat",
            json={"message": "Hola", "stream": True},
            headers={"Authorization": f"Bearer {token}"},
        ) as resp:
            assert resp.status_code == 200
            chunks = []
            async for line in resp.aiter_lines():
                if line.startswith("data: "):
                    payload = line[6:]
                    if payload != "[DONE]":
                        chunks.append(jsonlib.loads(payload))

    deltas = [c["delta"] for c in chunks if "delta" in c]
    assert deltas == ["Hola ", "desde ", "streaming "]
    final = chunks[-1]
    assert final["conversation_id"]
    assert final["tokens"] == {"input": 100, "output": 50}


async def test_assistant_history_and_clear(client, db_session):
    tenant = _create_tenant(db_session)
    _, token = _create_user(db_session, tenant, UserRole.EDITOR)

    # Seed una conversación
    from app.models.assistant import AssistantConversation

    conv = AssistantConversation(
        tenant_id=tenant.id, user_id="u1",
        messages=[{"role": "user", "content": "Hola", "ts": "now"}],
    )
    db_session.add(conv)
    db_session.commit()

    response = await client.get(
        "/api/v1/nlp/assistant/history",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    assert len(response.json()) == 1
    assert response.json()[0]["messages"][0]["content"] == "Hola"

    response = await client.delete(
        "/api/v1/nlp/assistant/history",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200

    response = await client.get(
        "/api/v1/nlp/assistant/history",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.json() == []
