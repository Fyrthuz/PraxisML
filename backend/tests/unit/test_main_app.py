from unittest.mock import patch, MagicMock
import pytest


class TestCreateApp:
    def test_create_app_returns_fastapi_app(self):
        with patch("app.main.setup_logging"), \
             patch("app.main._start_mlflow_ui"), \
             patch("app.main.Instrumentator") as mock_inst, \
             patch("app.main.RedirectResponse"), \
             patch("app.main.CORSMiddleware"), \
             patch("app.main.RequestIDMiddleware"):

            mock_instrumentor = MagicMock()
            mock_inst.return_value.instrument.return_value = mock_instrumentor

            from app.main import create_app
            app = create_app()
            assert app is not None
            assert app.title == "PraxisML"

    def test_create_app_sets_openapi_url(self):
        with patch("app.main.setup_logging"), \
             patch("app.main._start_mlflow_ui"), \
             patch("app.main.Instrumentator") as mock_inst, \
             patch("app.main.RedirectResponse"), \
             patch("app.main.CORSMiddleware"), \
             patch("app.main.RequestIDMiddleware"):

            mock_instrumentor = MagicMock()
            mock_inst.return_value.instrument.return_value = mock_instrumentor

            from app.main import create_app
            app = create_app()
            assert "/api/v1/openapi.json" in app.openapi_url

    def test_create_app_has_routes(self):
        with patch("app.main.setup_logging"), \
             patch("app.main._start_mlflow_ui"), \
             patch("app.main.Instrumentator") as mock_inst, \
             patch("app.main.RedirectResponse"), \
             patch("app.main.CORSMiddleware"), \
             patch("app.main.RequestIDMiddleware"):

            mock_instrumentor = MagicMock()
            mock_inst.return_value.instrument.return_value = mock_instrumentor

            from app.main import create_app
            app = create_app()
            routes = [r.path for r in app.routes]
            assert "/health" in routes
            assert "/api/v1/auth/login" in routes
            assert "/api/v1/datasets/" in routes
