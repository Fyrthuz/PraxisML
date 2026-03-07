"""
Jerarquía de excepciones de dominio de Antigravity SaaS.

Uso:
    raise ModelNotFoundError(model_id="abc-123")
    raise StorageError("No se pudo subir el artefacto", detail=str(exc))

Los handlers globales en main.py convierten estas excepciones a respuestas
JSON estructuradas con { code, message, detail }.
"""

from typing import Any


# ── Base ──────────────────────────────────────────────────────────────────────

class AntigravityError(Exception):
    """Excepción base del proyecto. Heredar para todos los errores de dominio."""

    status_code: int = 500
    code: str = "INTERNAL_ERROR"

    def __init__(self, message: str = "Error interno del servidor", detail: Any = None):
        self.message = message
        self.detail = detail
        super().__init__(message)


# ── Auth / RBAC ───────────────────────────────────────────────────────────────

class AuthenticationError(AntigravityError):
    status_code = 401
    code = "AUTHENTICATION_ERROR"

    def __init__(self, message: str = "No autenticado", detail: Any = None):
        super().__init__(message, detail)


class PermissionDeniedError(AntigravityError):
    """El usuario no tiene el rol necesario para esta operación."""
    status_code = 403
    code = "PERMISSION_DENIED"

    def __init__(self, message: str = "Permisos insuficientes", detail: Any = None):
        super().__init__(message, detail)


class TenantNotFoundError(AntigravityError):
    status_code = 404
    code = "TENANT_NOT_FOUND"

    def __init__(self, tenant_id: str | None = None, detail: Any = None):
        msg = f"Tenant no encontrado" + (f": {tenant_id}" if tenant_id else "")
        super().__init__(msg, detail)


# ── Modelos ML ────────────────────────────────────────────────────────────────

class ModelNotFoundError(AntigravityError):
    status_code = 404
    code = "MODEL_NOT_FOUND"

    def __init__(self, model_id: str | None = None, detail: Any = None):
        msg = "Modelo no encontrado" + (f": {model_id}" if model_id else "")
        super().__init__(msg, detail)


class ModelLoadError(AntigravityError):
    status_code = 500
    code = "MODEL_LOAD_ERROR"

    def __init__(self, message: str = "Error al cargar el modelo", detail: Any = None):
        super().__init__(message, detail)


class TrainingError(AntigravityError):
    status_code = 500
    code = "TRAINING_ERROR"

    def __init__(self, message: str = "Error durante el entrenamiento", detail: Any = None):
        super().__init__(message, detail)


class TrainingJobNotFoundError(AntigravityError):
    status_code = 404
    code = "TRAINING_JOB_NOT_FOUND"

    def __init__(self, task_id: str | None = None, detail: Any = None):
        msg = "Tarea de entrenamiento no encontrada" + (f": {task_id}" if task_id else "")
        super().__init__(msg, detail)


# ── Inferencia ────────────────────────────────────────────────────────────────

class PredictionNotFoundError(AntigravityError):
    status_code = 404
    code = "PREDICTION_NOT_FOUND"

    def __init__(self, prediction_id: str | None = None, detail: Any = None):
        msg = "Predicción no encontrada" + (f": {prediction_id}" if prediction_id else "")
        super().__init__(msg, detail)


class InferenceError(AntigravityError):
    status_code = 500
    code = "INFERENCE_ERROR"

    def __init__(self, message: str = "Error durante la inferencia", detail: Any = None):
        super().__init__(message, detail)


# ── Datasets ──────────────────────────────────────────────────────────────────

class DatasetNotFoundError(AntigravityError):
    status_code = 404
    code = "DATASET_NOT_FOUND"

    def __init__(self, dataset_id: str | None = None, detail: Any = None):
        msg = "Dataset no encontrado" + (f": {dataset_id}" if dataset_id else "")
        super().__init__(msg, detail)


class DatasetValidationError(AntigravityError):
    status_code = 422
    code = "DATASET_VALIDATION_ERROR"

    def __init__(self, message: str = "Dataset no válido", detail: Any = None):
        super().__init__(message, detail)


# ── Almacenamiento ────────────────────────────────────────────────────────────

class StorageError(AntigravityError):
    status_code = 500
    code = "STORAGE_ERROR"

    def __init__(self, message: str = "Error de almacenamiento", detail: Any = None):
        super().__init__(message, detail)


class StorageObjectNotFoundError(AntigravityError):
    status_code = 404
    code = "STORAGE_OBJECT_NOT_FOUND"

    def __init__(self, key: str | None = None, detail: Any = None):
        msg = "Objeto no encontrado en almacenamiento" + (f": {key}" if key else "")
        super().__init__(msg, detail)


# ── Preprocesamiento ──────────────────────────────────────────────────────────

class PreprocessingError(AntigravityError):
    status_code = 500
    code = "PREPROCESSING_ERROR"

    def __init__(self, message: str = "Error de preprocesamiento", detail: Any = None):
        super().__init__(message, detail)


# ── Rate limiting (para usar desde el middleware) ─────────────────────────────

class RateLimitExceededError(AntigravityError):
    status_code = 429
    code = "RATE_LIMIT_EXCEEDED"

    def __init__(self, message: str = "Demasiadas solicitudes. Inténtalo más tarde.", detail: Any = None):
        super().__init__(message, detail)
