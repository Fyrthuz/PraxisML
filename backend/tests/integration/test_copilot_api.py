"""
Tests de integración para el router transversal /api/v1/copilot/*
y verificación de persistencia real de respuestas SSE en assistant_conversations.
"""

import json as jsonlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import app.models.assistant  # noqa: F401
import app.models.embedding  # noqa: F401
from app.core.security import create_access_token, get_password_hash
from app.database import get_db
from app.main import create_app
from app.models.assistant import AssistantConversation
from app.models.base import Base
from app.models.tenant import Tenant
from app.models.user import User, UserRole

_fastapi_app = create_app()
SQLITE_URL = "sqlite:///./test_copilot_api.db"


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


def _seed_user(db_session, role: str):
    import uuid

    tenant = Tenant(name=f"Copilot Tenant {uuid.uuid4().hex[:6]}")
    db_session.add(tenant)
    db_session.commit()
    db_session.refresh(tenant)

    user = User(
        email=f"cp_{role}_{uuid.uuid4().hex[:8]}@test.com",
        hashed_password=get_password_hash("pass"),
        full_name="Copilot User",
        tenant_id=tenant.id,
        role=role,
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return tenant, user, create_access_token(subject=user.id)


async def test_copilot_tools_rbac_filtering(client, db_session):
    _, _, viewer_token = _seed_user(db_session, UserRole.VIEWER)
    _, _, editor_token = _seed_user(db_session, UserRole.EDITOR)

    r_viewer = await client.get(
        "/api/v1/copilot/tools",
        headers={"Authorization": f"Bearer {viewer_token}"},
    )
    assert r_viewer.status_code == 200
    viewer_tools = {t["name"] for t in r_viewer.json()["tools"]}
    assert "models.list" in viewer_tools
    assert "tabular.list_algorithms" in viewer_tools
    assert "nlp.list_algorithms" in viewer_tools
    assert "models.promote_stage" not in viewer_tools

    r_editor = await client.get(
        "/api/v1/copilot/tools",
        headers={"Authorization": f"Bearer {editor_token}"},
    )
    assert r_editor.status_code == 200
    editor_tools = {t["name"] for t in r_editor.json()["tools"]}
    assert "models.promote_stage" in editor_tools


async def test_copilot_plugins_endpoint(client, db_session):
    _, _, viewer_token = _seed_user(db_session, UserRole.VIEWER)

    resp = await client.get(
        "/api/v1/copilot/plugins",
        headers={"Authorization": f"Bearer {viewer_token}"},
    )
    assert resp.status_code == 200
    names = {p["name"] for p in resp.json()["plugins"]}
    assert {"tabular", "nlp"}.issubset(names)


async def test_copilot_sse_persists_full_accumulated_answer(client, db_session):
    tenant, _, editor_token = _seed_user(db_session, UserRole.EDITOR)

    gateway = MagicMock()
    gateway.record_usage = MagicMock()

    async def _stream_gen(*args, **kwargs):
        yield {"delta": "Respuesta "}
        yield {"delta": "completa "}
        yield {"delta": "streamed."}
        yield {"usage": {"input": 42, "output": 18}}

    gateway.stream = _stream_gen
    gateway.complete = AsyncMock()

    rag = MagicMock()
    rag.SYSTEM_PROMPT = "Prompt"
    rag.retrieve_context.return_value = {"context": "{}", "sources": []}

    with patch("app.services.llm_gateway.get_llm_gateway", return_value=gateway), patch(
        "app.services.rag_service.get_rag_service", return_value=rag
    ):
        async with client.stream(
            "POST",
            "/api/v1/copilot/chat",
            json={"message": "Resume mis modelos", "stream": True},
            headers={"Authorization": f"Bearer {editor_token}"},
        ) as resp:
            assert resp.status_code == 200
            events = []
            async for line in resp.aiter_lines():
                if line.startswith("data: ") and line[6:] != "[DONE]":
                    events.append(jsonlib.loads(line[6:]))

    assert events[-1]["tokens"] == {"input": 42, "output": 18}
    conv = (
        db_session.query(AssistantConversation)
        .filter(AssistantConversation.tenant_id == tenant.id)
        .first()
    )
    assert conv is not None
    assert conv.messages[-1]["role"] == "assistant"
    assert conv.messages[-1]["content"] == "Respuesta completa streamed."
    gateway.record_usage.assert_called_once()
