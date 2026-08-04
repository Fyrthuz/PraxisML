"""
Tests de integración para los endpoints de observabilidad (Fase 0).

Ejecutar:
    cd backend
    uv run pytest tests/integration/test_observability_api.py -v
"""

import uuid

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.main import create_app
from app.database import get_db

_fastapi_app = create_app()
from app.models.base import Base
from app.models.user import User, UserRole
from app.models.tenant import Tenant
from app.core.security import create_access_token, get_password_hash

# Importar modelos de observabilidad para que Base.metadata los descubra
import app.observability.models  # noqa: F401

SQLITE_URL = "sqlite:///./test_observability.db"


@pytest.fixture(scope="module")
def engine():
    eng = create_engine(SQLITE_URL, connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=eng)
    yield eng
    Base.metadata.drop_all(bind=eng)


@pytest.fixture(autouse=True)
def _clean_db(engine):
    """Limpia todas las tablas antes de cada test."""
    with engine.connect() as conn:
        for table in reversed(Base.metadata.sorted_tables):
            conn.execute(table.delete())
        conn.commit()
    yield


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
    _fastapi_app.dependency_overrides[get_db] = lambda: db_session
    yield
    _fastapi_app.dependency_overrides.clear()


@pytest_asyncio.fixture()
async def client(override_db):
    async with AsyncClient(transport=ASGITransport(app=_fastapi_app), base_url="http://test") as ac:
        yield ac


# ── Helpers ──────────────────────────────────────────────────────────────


def _create_tenant(db_session, name="Test Tenant Observability") -> Tenant:
    tenant = Tenant(name=name)
    db_session.add(tenant)
    db_session.commit()
    db_session.refresh(tenant)
    return tenant


def _create_user(db_session, tenant: Tenant, role: str) -> tuple[User, str]:
    email = f"obs_{role}_{uuid.uuid4().hex[:8]}@test.com"
    user = User(
        email=email,
        hashed_password=get_password_hash("pass"),
        full_name=f"Test {role.capitalize()}",
        tenant_id=tenant.id,
        role=role,
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    token = create_access_token(subject=user.id)
    return user, token


# ── Tests ────────────────────────────────────────────────────────────────


class TestObservabilityHealth:
    async def test_health_unauthenticated(self, client):
        """GET /health debe funcionar sin auth."""
        response = await client.get("/health")
        assert response.status_code == 200


class TestObservabilityAuth:
    """Verifica que los endpoints requieran auth correctamente."""

    @pytest_asyncio.fixture()
    async def setup(self, db_session, override_db):
        tenant = _create_tenant(db_session)
        admin_user, admin_token = _create_user(db_session, tenant, UserRole.ADMIN)
        editor_user, editor_token = _create_user(db_session, tenant, UserRole.EDITOR)
        viewer_user, viewer_token = _create_user(db_session, tenant, UserRole.VIEWER)
        return {
            "tenant": tenant,
            "admin_token": admin_token,
            "editor_token": editor_token,
            "viewer_token": viewer_token,
        }

    async def test_usage_requires_auth(self, client, setup):
        response = await client.get("/api/v1/observability/usage")
        assert response.status_code == 401

    async def test_usage_denies_viewer(self, client, setup):
        """Usage requiere role editor, viewer debe recibir 403."""
        headers = {"Authorization": f"Bearer {setup['viewer_token']}"}
        response = await client.get("/api/v1/observability/usage", headers=headers)
        assert response.status_code == 403

    async def test_pricing_list_allows_editor(self, client, setup):
        headers = {"Authorization": f"Bearer {setup['editor_token']}"}
        response = await client.get("/api/v1/observability/pricing", headers=headers)
        assert response.status_code == 200

    async def test_pricing_create_requires_admin(self, client, setup):
        # Editor no puede crear precios
        headers = {"Authorization": f"Bearer {setup['editor_token']}"}
        response = await client.post("/api/v1/observability/pricing", json={}, headers=headers)
        assert response.status_code == 403

        # Admin sí puede
        headers = {"Authorization": f"Bearer {setup['admin_token']}"}
        response = await client.post(
            "/api/v1/observability/pricing",
            json={
                "resource_category": "compute_gpu",
                "resource_id": "H100",
                "pricing_unit": "per_hour",
                "price_per_unit": 4.50,
            },
            headers=headers,
        )
        assert response.status_code == 200

    async def test_alerts_crud(self, client, setup):
        headers = {"Authorization": f"Bearer {setup['editor_token']}"}

        # Crear alerta
        response = await client.post(
            "/api/v1/observability/alerts",
            json={
                "alert_type": "cost_spike",
                "metric_name": "inference_cost_usd",
                "threshold": 2.0,
                "comparison": "pct_change",
                "channels": [{"type": "webhook", "url": "https://hooks.example.com"}],
            },
            headers=headers,
        )
        assert response.status_code == 201
        alert_id = response.json()["id"]

        # Listar alertas
        response = await client.get("/api/v1/observability/alerts", headers=headers)
        assert response.status_code == 200
        assert len(response.json()) == 1

        # Actualizar alerta
        response = await client.patch(
            f"/api/v1/observability/alerts/{alert_id}",
            json={"enabled": False},
            headers=headers,
        )
        assert response.status_code == 200
        assert response.json()["enabled"] is False

        # Eliminar alerta (solo admin)
        admin_headers = {"Authorization": f"Bearer {setup['admin_token']}"}
        response = await client.delete(
            f"/api/v1/observability/alerts/{alert_id}",
            headers=admin_headers,
        )
        assert response.status_code == 204

    async def test_pricing_history_requires_admin(self, client, setup):
        headers = {"Authorization": f"Bearer {setup['viewer_token']}"}
        response = await client.get("/api/v1/observability/pricing/history", headers=headers)
        assert response.status_code == 403

    async def test_model_health_allows_viewer(self, client, setup):
        headers = {"Authorization": f"Bearer {setup['viewer_token']}"}
        response = await client.get(
            "/api/v1/observability/models/nonexistent/health", headers=headers
        )
        assert response.status_code == 404

    async def test_delete_alerts_requires_admin(self, client, setup):
        # Editor no puede eliminar
        headers = {"Authorization": f"Bearer {setup['editor_token']}"}
        response = await client.delete(
            "/api/v1/observability/alerts/fake-id", headers=headers
        )
        assert response.status_code == 403

        # Admin con ID inexistente da 404
        admin_headers = {"Authorization": f"Bearer {setup['admin_token']}"}
        response = await client.delete(
            "/api/v1/observability/alerts/nonexistent", headers=admin_headers
        )
        assert response.status_code == 404


class TestObservabilityDataFlow:
    """Prueba el flujo completo: métricas → costes → drift → health."""

    @pytest_asyncio.fixture()
    async def setup(self, db_session, override_db):
        tenant = _create_tenant(db_session, "Data Flow Tenant")
        _, token = _create_user(db_session, tenant, UserRole.ADMIN)

        # Sembrar datos de prueba
        from app.observability.models import UsageMetric, CostRecord, DriftHistory

        now = __import__("datetime").datetime.now(timezone := __import__("datetime").timezone.utc)

        # Métricas de inferencia
        for i in range(5):
            db_session.add(UsageMetric(
                tenant_id=tenant.id, model_id="model-test-1", domain="tabular",
                metric_name="inference_latency_ms", metric_value=100 + i * 10,
                tags={"algorithm": "rf"}, recorded_at=now,
            ))
        # Costes
        db_session.add(CostRecord(
            tenant_id=tenant.id, model_id="model-test-1", cost_type="inference",
            amount=0.25, source="pred-1", period_start=now, period_end=now,
        ))
        db_session.add(CostRecord(
            tenant_id=tenant.id, model_id="model-test-1", cost_type="training",
            amount=5.0, source="train-1", period_start=now, period_end=now,
        ))
        # Drift
        db_session.add(DriftHistory(
            model_id="model-test-1", tenant_id=tenant.id, drift_type="data_drift",
            metrics={"psi": 0.12, "ks": 0.04}, recorded_at=now,
        ))
        db_session.commit()

        return {"tenant": tenant, "token": token}

    async def test_get_usage(self, client, setup):
        headers = {"Authorization": f"Bearer {setup['token']}"}
        response = await client.get(
            "/api/v1/observability/usage?metric=inference_latency_ms", headers=headers
        )
        assert response.status_code == 200
        data = response.json()
        assert data["tenant_id"] == setup["tenant"].id
        assert len(data["metrics"]) > 0

    async def test_get_usage_by_model(self, client, setup):
        headers = {"Authorization": f"Bearer {setup['token']}"}
        response = await client.get(
            "/api/v1/observability/usage/models/model-test-1", headers=headers
        )
        assert response.status_code == 200
        data = response.json()
        assert data["model_id"] == "model-test-1"

    async def test_get_costs(self, client, setup):
        headers = {"Authorization": f"Bearer {setup['token']}"}
        response = await client.get("/api/v1/observability/costs", headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert data["total_cost_usd"] > 0
        assert len(data["records"]) == 2

    async def test_get_cost_forecast(self, client, setup):
        headers = {"Authorization": f"Bearer {setup['token']}"}
        response = await client.get(
            "/api/v1/observability/costs/forecast", headers=headers
        )
        assert response.status_code == 200

    async def test_get_pricing_active(self, client, setup):
        headers = {"Authorization": f"Bearer {setup['token']}"}
        response = await client.get("/api/v1/observability/pricing", headers=headers)
        assert response.status_code == 200
        # La tabla está vacía en test (no hay seed)
        assert isinstance(response.json(), list)

    async def test_create_and_verify_pricing(self, client, setup):
        headers = {"Authorization": f"Bearer {setup['token']}"}
        # Crear precio
        response = await client.post(
            "/api/v1/observability/pricing",
            json={
                "resource_category": "compute_gpu",
                "resource_id": "H100",
                "pricing_unit": "per_hour",
                "price_per_unit": 4.50,
            },
            headers=headers,
        )
        assert response.status_code == 200
        pid = response.json()["id"]
        assert pid is not None

        # Verificar que aparece en listado
        response = await client.get(
            "/api/v1/observability/pricing?category=compute_gpu", headers=headers
        )
        assert response.status_code == 200
        items = response.json()
        assert any(item["resource_id"] == "H100" for item in items)

        # Verificar historial
        response = await client.get(
            "/api/v1/observability/pricing/history?category=compute_gpu",
            headers={"Authorization": f"Bearer {setup['token']}"},
        )
        assert response.status_code == 200

    async def test_get_drift_history(self, client, setup):
        headers = {"Authorization": f"Bearer {setup['token']}"}
        response = await client.get(
            "/api/v1/observability/drift/history/model-test-1", headers=headers
        )
        assert response.status_code == 200
        data = response.json()
        assert data["model_id"] == "model-test-1"
        assert len(data["drift_points"]) == 1
        assert data["drift_points"][0]["psi"] == 0.12

    async def test_model_health(self, client, setup):
        headers = {"Authorization": f"Bearer {setup['token']}"}
        # Sin modelo en la DB — debería dar 404
        response = await client.get(
            "/api/v1/observability/models/model-test-1/health", headers=headers
        )
        assert response.status_code == 404

        # Crear un modelo y verificar health
        from app.models.ml_model import MLModel

        db_session = None
        for dep_key, dep_val in _fastapi_app.dependency_overrides.items():
            from app.database import get_db as get_db_fn
            if dep_key == get_db_fn:
                db_session = dep_val()
                break

        if db_session:
            model = MLModel(
                id="model-health-1",
                name="Health Test Model",
                mlflow_run_id="run-health-1",
                metrics_metadata={"f1": 0.89, "accuracy": 0.91},
                tenant_id=setup["tenant"].id,
                stage="Production",
            )
            db_session.add(model)
            db_session.commit()

            response = await client.get(
                "/api/v1/observability/models/model-health-1/health", headers=headers
            )
            assert response.status_code == 200
            data = response.json()
            assert data["health_score"] is not None
            assert len(data["components"]) == 3
            # accuracy debe ser ~90 basado en f1=0.89
            accuracy_comp = [c for c in data["components"] if c["name"] == "accuracy"]
            assert len(accuracy_comp) == 1


class TestTenantBudget:
    """Prueba los campos de budget en la API de tenants."""

    @pytest_asyncio.fixture()
    async def setup(self, db_session, override_db):
        tenant = _create_tenant(db_session, "Budget Tenant")
        _, token = _create_user(db_session, tenant, UserRole.ADMIN)
        return {"tenant": tenant, "token": token}

    async def test_tenant_response_has_budget_fields(self, client, setup):
        headers = {"Authorization": f"Bearer {setup['token']}"}
        response = await client.get("/api/v1/tenants/me", headers=headers)
        assert response.status_code == 200
        data = response.json()
        # Debería tener los campos de budget aunque sean null
        assert "monthly_budget_usd" in data
        assert "budget_alert_pct" in data
        assert "billing_cycle" in data

    async def test_tenant_update_budget(self, client, setup):
        headers = {"Authorization": f"Bearer {setup['token']}"}
        tenant_id = setup["tenant"].id

        response = await client.patch(
            f"/api/v1/tenants/{tenant_id}",
            json={"monthly_budget_usd": 1000.0, "budget_alert_pct": 85.0},
            headers=headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["monthly_budget_usd"] == 1000.0
        assert data["budget_alert_pct"] == 85.0

        # Verificar persistencia
        response = await client.get("/api/v1/tenants/me", headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert data["monthly_budget_usd"] == 1000.0


class TestUsageCollectionFlow:
    """Prueba el uso del UsageCollector como dependencia."""

    async def test_usage_collector_importable(self):
        from app.observability.usage_collector import get_usage_collector, UsageCollector

        collector = get_usage_collector()
        assert isinstance(collector, UsageCollector)

    async def test_cost_calculator_importable(self):
        from app.observability.cost_calculator import get_cost_calculator, CostCalculator

        calc = get_cost_calculator()
        assert isinstance(calc, CostCalculator)

    async def test_alert_engine_importable(self):
        from app.observability.alert_engine import get_alert_engine, AlertEngine

        engine = get_alert_engine()
        assert isinstance(engine, AlertEngine)

    async def test_reports_importable(self):
        from app.observability.reports import generate_cost_csv, generate_monthly_report_pdf

        assert callable(generate_cost_csv)
        assert callable(generate_monthly_report_pdf)


class TestObservabilityOpenAPI:
    """Verifica que los endpoints aparezcan en OpenAPI."""

    async def test_observability_paths_in_openapi(self, client):
        response = await client.get("/api/v1/openapi.json")
        assert response.status_code == 200
        paths = response.json().get("paths", {})
        obs_paths = {k: v for k, v in paths.items() if "observability" in k}
        assert len(obs_paths) >= 10, f"Solo {len(obs_paths)} rutas observability encontradas"
        expected = [
            "/api/v1/observability/usage",
            "/api/v1/observability/usage/models/{model_id}",
            "/api/v1/observability/costs",
            "/api/v1/observability/costs/forecast",
            "/api/v1/observability/pricing",
            "/api/v1/observability/pricing/history",
            "/api/v1/observability/drift/history/{model_id}",
            "/api/v1/observability/alerts",
            "/api/v1/observability/alerts/{alert_id}",
            "/api/v1/observability/models/{model_id}/health",
            "/api/v1/observability/tenant/{tenant_id}/report",
        ]
        for path in expected:
            assert path in obs_paths, f"Falta ruta: {path}"
