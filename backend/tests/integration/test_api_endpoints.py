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
from app.models.user import User, UserRole
from app.models.tenant import Tenant
from app.core.security import create_access_token, get_password_hash

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


# ── Helpers: crear usuarios de test con roles ─────────────────────────────────

def _create_test_tenant(db_session, name="Test Tenant RBAC") -> Tenant:
    """Crea un tenant de test y lo devuelve."""
    tenant = Tenant(name=name)
    db_session.add(tenant)
    db_session.commit()
    db_session.refresh(tenant)
    return tenant


def _create_test_user(db_session, tenant: Tenant, role: str, email_suffix: str = "") -> tuple[User, str]:
    """Crea un usuario de test con un rol y devuelve (user, token)."""
    email = f"test_{role}{email_suffix}@rbac.test"
    user = User(
        email=email,
        hashed_password=get_password_hash("testpass123"),
        full_name=f"Test {role.capitalize()}",
        tenant_id=tenant.id,
        role=role,
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    token = create_access_token(subject=user.id)
    return user, token


# ── Health ────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_health_check(client):
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


# ── Auth ──────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_register_creates_admin_user(client, db_session):
    """El primer usuario registrado debe obtener rol 'admin'."""
    response = await client.post(
        "/api/v1/auth/register",
        json={
            "email": "newadmin@test.com",
            "password": "securepass123",
            "full_name": "New Admin",
            "tenant_name": "My Company",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["role"] == "admin"


@pytest.mark.asyncio
async def test_login_invalid_credentials(client):
    """Login con credenciales incorrectas debe devolver error."""
    response = await client.post(
        "/api/v1/auth/login",
        data={"username": "noexiste@test.com", "password": "wrong"},
    )
    assert response.status_code in (400, 401, 422)


# ── RBAC: Models ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_list_models_unauthenticated(client):
    """Sin token la API debe devolver 401."""
    response = await client.get("/api/v1/models/")
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_list_models_viewer_allowed(client, db_session):
    """Viewer puede listar modelos (lectura)."""
    tenant = _create_test_tenant(db_session, "Models Viewer Tenant")
    _, token = _create_test_user(db_session, tenant, UserRole.VIEWER, "_models_list")
    response = await client.get(
        "/api/v1/models/",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_register_model_viewer_forbidden(client, db_session):
    """Viewer NO puede registrar modelos (requiere editor)."""
    tenant = _create_test_tenant(db_session, "Models Viewer Forbidden")
    _, token = _create_test_user(db_session, tenant, UserRole.VIEWER, "_models_reg")
    response = await client.post(
        "/api/v1/models/",
        json={
            "name": "Test Model",
            "mlflow_run_id": "fake-run-123",
        },
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 403


@pytest.mark.asyncio
async def test_delete_model_editor_forbidden(client, db_session):
    """Editor NO puede borrar modelos (requiere admin)."""
    tenant = _create_test_tenant(db_session, "Models Editor Forbidden")
    _, token = _create_test_user(db_session, tenant, UserRole.EDITOR, "_models_del")
    response = await client.delete(
        "/api/v1/models/nonexistent-id",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 403


# ── RBAC: Datasets ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_list_datasets_viewer_ok(client, db_session):
    """Viewer puede listar datasets."""
    tenant = _create_test_tenant(db_session, "DS Viewer Tenant")
    _, token = _create_test_user(db_session, tenant, UserRole.VIEWER, "_ds_list")
    response = await client.get(
        "/api/v1/datasets/",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_delete_dataset_editor_forbidden(client, db_session):
    """Editor NO puede borrar datasets (requiere admin)."""
    tenant = _create_test_tenant(db_session, "DS Editor Forbidden")
    _, token = _create_test_user(db_session, tenant, UserRole.EDITOR, "_ds_del")
    response = await client.delete(
        "/api/v1/datasets/nonexistent-id",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 403


# ── RBAC: Predictions ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_list_predictions_unauthenticated(client):
    """Sin token la API debe devolver 401."""
    response = await client.get("/api/v1/predictions")
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_list_predictions_viewer_ok(client, db_session):
    """Viewer puede listar predicciones."""
    tenant = _create_test_tenant(db_session, "Pred Viewer Tenant")
    _, token = _create_test_user(db_session, tenant, UserRole.VIEWER, "_pred_list")
    response = await client.get(
        "/api/v1/predictions",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200


# ── RBAC: Training ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_training_unauthenticated(client):
    """Sin token el endpoint de entrenamiento debe devolver 401."""
    response = await client.post("/api/v1/training/train", json={})
    assert response.status_code in (401, 422)


@pytest.mark.asyncio
async def test_training_viewer_forbidden(client, db_session):
    """Viewer NO puede lanzar entrenamientos (requiere editor)."""
    tenant = _create_test_tenant(db_session, "Training Viewer Forbidden")
    _, token = _create_test_user(db_session, tenant, UserRole.VIEWER, "_train_viewer")
    response = await client.post(
        "/api/v1/training/train",
        json={
            "dataset_id": "fake-ds-id",
            "target_column": "target",
            "algorithm": "random_forest",
        },
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 403


@pytest.mark.asyncio
async def test_algorithms_viewer_ok(client, db_session):
    """Viewer puede ver la lista de algoritmos."""
    tenant = _create_test_tenant(db_session, "Algo Viewer Tenant")
    _, token = _create_test_user(db_session, tenant, UserRole.VIEWER, "_algo_list")
    response = await client.get(
        "/api/v1/training/algorithms",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200


# ── RBAC: Tenants ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_create_tenant_viewer_forbidden(client, db_session):
    """Viewer NO puede crear tenants (requiere admin)."""
    tenant = _create_test_tenant(db_session, "Tenant Viewer Forbidden")
    _, token = _create_test_user(db_session, tenant, UserRole.VIEWER, "_tenant_create")
    response = await client.post(
        "/api/v1/tenants/",
        json={"name": "New Tenant"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 403


@pytest.mark.asyncio
async def test_create_tenant_editor_forbidden(client, db_session):
    """Editor NO puede crear tenants (requiere admin)."""
    tenant = _create_test_tenant(db_session, "Tenant Editor Forbidden")
    _, token = _create_test_user(db_session, tenant, UserRole.EDITOR, "_tenant_create")
    response = await client.post(
        "/api/v1/tenants/",
        json={"name": "New Tenant"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 403


# ── Error handler ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_404_returns_json(client):
    """Las rutas no existentes deben devolver JSON, no HTML."""
    response = await client.get("/ruta-que-no-existe")
    assert response.status_code == 404
    # FastAPI devuelve JSON por defecto para 404
    assert response.headers.get("content-type", "").startswith("application/json")
