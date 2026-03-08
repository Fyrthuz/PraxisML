"""
Logging estructurado JSON para PraxisML.

Configuración:
    from app.core.logging import setup_logging
    setup_logging()                # llamar al inicio de la app

Uso en módulos:
    import logging
    logger = logging.getLogger(__name__)
    logger.info("Mensaje", extra={"tenant_id": "t1", "model_id": "m42"})

Campos automáticos en cada registro JSON:
    timestamp, level, logger, message,
    tenant_id, model_id, task_id, request_id
"""

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any


PRAXISML_FIELDS = ("tenant_id", "model_id", "task_id", "request_id", "user_id")


class JsonFormatter(logging.Formatter):
    """Formatea cada log como una línea JSON con campos estándar."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Añadir campos de contexto si están presentes en el `extra`
        for field in PRAXISML_FIELDS:
            value = getattr(record, field, None)
            if value is not None:
                log_entry[field] = value

        # Añadir información de excepción si existe
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry, ensure_ascii=False)


def setup_logging(level: str = "INFO") -> None:
    """
    Configura el sistema de logging global.
    - Producción / contenedor → JSON por stdout.
    - Desarrollo local → mismo formato JSON (facilita parseo con herramientas).
    Llamar una única vez al arranque de la aplicación.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Evitar duplicar handlers si se llama varias veces (tests, reloads)
    if root_logger.handlers:
        root_logger.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root_logger.addHandler(handler)

    # Silenciar loggers muy verbosos de librerías externas
    for noisy in (
        "uvicorn.access",
        "sqlalchemy.engine",
        "httpx",
        "httpcore",
        "multipart",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Shortcut para obtener un logger con el nombre del módulo."""
    return logging.getLogger(name)
