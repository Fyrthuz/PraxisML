import logging
import os
import subprocess
import sys
from contextlib import asynccontextmanager

os.environ["GIT_PYTHON_REFRESH"] = "quiet"

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from prometheus_client import Counter
from prometheus_fastapi_instrumentator import Instrumentator
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.core.config import settings
from app.core.exceptions import PraxisMLError
from app.core.logging import setup_logging
from app.core.middleware import RequestIDMiddleware
from app.core.rate_limit import limiter

# ── Logging estructurado ───────────────────────────────────────────────────────
setup_logging(level="DEBUG" if settings.ENVIRONMENT == "development" else "INFO")
logger = logging.getLogger(__name__)

# ── Métricas Custom ────────────────────────────────────────────────────────────
api_errors_total = Counter(
    "api_errors_total",
    "Total de errores capturados por la API",
    ["error_type", "handler", "status_code"],
)

# ── MLFlow UI subprocess (local dev only) ─────────────────────────────────────
_mlflow_proc: subprocess.Popen | None = None

MLFLOW_UI_PORT = 5001


def _start_mlflow_ui() -> None:
    """
    Lanza `mlflow ui` en background en el puerto MLFLOW_UI_PORT.
    Si MLFlow no está disponible o ya hay algo corriendo en ese puerto,
    lo registra y sigue (no es crítico para la API).
    """
    global _mlflow_proc
    from app.services.mlflow_service import MLFlowService

    try:
        tracking_uri = MLFlowService().get_tracking_uri()
        # Si el tracking URI es remoto (http), no iniciamos el proceso local
        if tracking_uri.startswith("http"):
            logger.info(
                "MLFlow tracking URI es remoto (%s), saltando MLFlow UI local.",
                tracking_uri,
            )
            return

        _mlflow_proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "mlflow",
                "ui",
                "--backend-store-uri",
                tracking_uri,
                "--port",
                str(MLFLOW_UI_PORT),
                "--host",
                "0.0.0.0",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        logger.info(
            "MLFlow UI iniciado en http://localhost:%s (PID %s)",
            MLFLOW_UI_PORT,
            _mlflow_proc.pid,
        )
    except Exception as exc:
        logger.warning("No se pudo iniciar MLFlow UI: %s", exc)


def _stop_mlflow_ui() -> None:
    global _mlflow_proc
    if _mlflow_proc and _mlflow_proc.poll() is None:
        _mlflow_proc.terminate()
        logger.info("MLFlow UI detenido.")


# ── App lifecycle ─────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    _start_mlflow_ui()
    yield
    _stop_mlflow_ui()


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.PROJECT_NAME,
        openapi_url=f"{settings.API_V1_STR}/openapi.json",
        description="PraxisML — Plataforma de ML Engineering & AI",
        version="0.1.0",
        lifespan=lifespan,
    )

    # ── Rate Limiter state ────────────────────────────────────────────────────
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # ── Request ID — inyecta X-Request-ID en cada request/response ──────────
    app.add_middleware(RequestIDMiddleware)

    # ── CORS — restringido al dominio del frontend por entorno ────────────────
    # Para desarrollo, permitir todos los orígenes si es necesario
    cors_origins = settings.cors_origins_list
    if settings.ENVIRONMENT == "development":
        # En desarrollo, permitir cualquier origen para facilitar pruebas
        # Incluir explícitamente el origen del frontend para WebSockets
        cors_origins = [
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "ws://localhost:3000",
            "ws://127.0.0.1:3000",
            "*",
        ]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Middleware de debug para WebSockets en desarrollo
    if settings.ENVIRONMENT == "development":

        @app.middleware("http")
        async def debug_websocket_middleware(request, call_next):
            if request.url.path.startswith("/api/v1/streaming"):
                logger.info(
                    f"WebSocket debug: Origin={request.headers.get('origin')}, Host={request.headers.get('host')}, URL={request.url}"
                )
            response = await call_next(request)
            return response

    # ── Global exception handlers ─────────────────────────────────────────────

    @app.exception_handler(PraxisMLError)
    async def praxisml_error_handler(request: Request, exc: PraxisMLError):
        logger.warning(
            "PraxisMLError [%s]: %s",
            exc.code,
            exc.message,
            extra={"detail": str(exc.detail) if exc.detail else None},
        )
        api_errors_total.labels(
            error_type="PraxisMLError",
            handler=request.url.path,
            status_code=exc.status_code,
        ).inc()
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "code": exc.code,
                "message": exc.message,
                "detail": exc.detail,
            },
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        logger.exception(
            "Error no controlado en %s %s", request.method, request.url.path
        )
        api_errors_total.labels(
            error_type="UnhandledException", handler=request.url.path, status_code=500
        ).inc()
        return JSONResponse(
            status_code=500,
            content={
                "code": "INTERNAL_ERROR",
                "message": "Error interno del servidor",
                "detail": str(exc) if settings.ENVIRONMENT == "development" else None,
            },
        )

    # ── Prometheus metrics (/metrics) ─────────────────────────────────────────
    Instrumentator().instrument(app).expose(app, endpoint="/metrics", tags=["System"])

    # ── Health ────────────────────────────────────────────────────────────────
    @app.get("/health", tags=["System"])
    def health_check():
        return {
            "status": "ok",
            "project": settings.PROJECT_NAME,
            "env": settings.ENVIRONMENT,
        }

    # ── MLFlow UI redirect ────────────────────────────────────────────────────
    @app.get("/mlflow", tags=["System"], include_in_schema=False)
    def mlflow_ui_redirect():
        """Redirige al dashboard de MLFlow UI."""
        tracking_uri = settings.MLFLOW_TRACKING_URI
        if tracking_uri and tracking_uri.startswith("http"):
            return RedirectResponse(url=tracking_uri)
        return RedirectResponse(url=f"http://localhost:{MLFLOW_UI_PORT}")

    # ── Routers ───────────────────────────────────────────────────────────────
    from app.api.routes.v1 import (
        auth,
        datasets,
        drift,
        models,
        predictions,
        preprocessing,
        profiling,
        streaming,
        tenants,
        training,
        users,
    )

    app.include_router(auth.router, prefix=f"{settings.API_V1_STR}/auth", tags=["Auth"])
    app.include_router(
        tenants.router, prefix=f"{settings.API_V1_STR}/tenants", tags=["Tenants"]
    )
    app.include_router(
        datasets.router, prefix=f"{settings.API_V1_STR}/datasets", tags=["Datasets"]
    )
    app.include_router(
        profiling.router, prefix=f"{settings.API_V1_STR}/profiling", tags=["Profiling"]
    )
    app.include_router(
        models.router, prefix=f"{settings.API_V1_STR}/models", tags=["Models"]
    )
    app.include_router(
        predictions.router, prefix=f"{settings.API_V1_STR}", tags=["Predictions"]
    )
    app.include_router(
        preprocessing.router,
        prefix=f"{settings.API_V1_STR}/preprocessing",
        tags=["Preprocessing"],
    )
    app.include_router(
        training.router, prefix=f"{settings.API_V1_STR}/training", tags=["Training"]
    )
    app.include_router(
        streaming.router, prefix=f"{settings.API_V1_STR}", tags=["Streaming"]
    )
    app.include_router(drift.router, prefix=f"{settings.API_V1_STR}", tags=["Drift"])
    app.include_router(
        users.router, prefix=f"{settings.API_V1_STR}/users", tags=["Users"]
    )

    return app


app = create_app()
