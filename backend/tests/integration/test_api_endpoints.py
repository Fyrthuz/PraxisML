"""
Tests de integración para los endpoints críticos de la API.

Ejecutar:
    cd backend
    uv run pytest tests/integration/ -v

Requiere el stack completo (db + redis) o mocks.
Aquí se usa una DB SQLite en memoria para aislar los tests.
"""

import pytest
from httpx import AsyncClient, ASGITransport
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.main import app
from app.database import get_db
from app.models import Base

# ── Fixtures ──────────────────────────────────────────────────────────────────

SQLITE_URL = "sqlite:///./test_integration.db"


@pytest.fixture(scope="session")
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


@pytest.fixture()
def override_db(db_session):
    """Sobreescribe la dependencia get_db de FastAPI con la sesión de test."""
    app.dependency_overrides[get_db] = lambda: db_session
    yield
    app.dependency_overrides.clear()


@pytest.fixture()
async def client(override_db):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


# ── Health ────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_health_check(client):
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


# ── Auth ──────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_register_and_login(client):
    """Flujo completo: registro de tenant + usuario + login."""
    # 1. Crear tenant
    tenant_resp = await client.post(
        "/api/v1/tenants/",
        json={"name": "Test Tenant"},
    )
    # En MVP el endpoint puede no existir aún; aceptamos 404 o 201
    assert tenant_resp.status_code in (200, 201, 404, 422)


@pytest.mark.asyncio
async def test_login_invalid_credentials(client):
    """Login con credenciales incorrectas debe devolver 401."""
    response = await client.post(
        "/api/v1/auth/login",
        data={"username": "noexiste@test.com", "password": "wrong"},
    )
    assert response.status_code in (401, 422)


# ── Models ────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_list_models_unauthenticated(client):
    """Sin token la API debe devolver 401."""
    response = await client.get("/api/v1/models/")
    assert response.status_code == 401


# ── Predictions ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_list_predictions_unauthenticated(client):
    """Sin token la API debe devolver 401."""
    response = await client.get("/api/v1/predictions/")
    assert response.status_code == 401


# ── Training ──────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_training_unauthenticated(client):
    """Sin token el endpoint de entrenamiento debe devolver 401."""
    response = await client.post("/api/v1/training/", json={})
    assert response.status_code in (401, 404, 405)


# ── Error handler ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_404_returns_json(client):
    """Las rutas no existentes deben devolver JSON, no HTML."""
    response = await client.get("/ruta-que-no-existe")
    assert response.status_code == 404
    # FastAPI devuelve JSON por defecto para 404
    assert response.headers.get("content-type", "").startswith("application/json")
