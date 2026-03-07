"""
StorageService — Interfaz abstracta de almacenamiento de artefactos.

Implementaciones disponibles (seleccionables por STORAGE_BACKEND env var):
    - "local"  → disco local (default para desarrollo sin MinIO)
    - "minio"  → MinIO self-hosted
    - "s3"     → AWS S3

Uso:
    from app.services.storage_service import get_storage
    storage = get_storage()
    storage.upload("tenant_id/predictions/result.npy", data_bytes)
    url = storage.get_url("tenant_id/predictions/result.npy")
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import BinaryIO


class StorageService(ABC):
    """Interfaz abstracta. Cada backend implementa estos tres métodos."""

    @abstractmethod
    def upload(self, key: str, data: bytes | BinaryIO, content_type: str = "application/octet-stream") -> str:
        """
        Sube dato/fichero al backend.
        Retorna la object key (misma que la de entrada, para simpliidad).
        """
        ...

    @abstractmethod
    def download(self, key: str) -> bytes:
        """Descarga el objeto y lo retorna como bytes. Lanza StorageObjectNotFoundError si no existe."""
        ...

    @abstractmethod
    def get_url(self, key: str, expires_in: int = 3600) -> str:
        """
        Retorna una URL pre-firmada o pública para acceder al objeto.
        expires_in: segundos de validez (para backends que soporten URLs firmadas).
        """
        ...

    @abstractmethod
    def delete(self, key: str) -> None:
        """Elimina el objeto. No lanza error si no existía."""
        ...

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Comprueba si un objeto existe."""
        ...


# ── Factory ───────────────────────────────────────────────────────────────────

def get_storage() -> StorageService:
    """
    Retorna la implementación de StorageService según STORAGE_BACKEND.
    Se instancia como singleton ligero (sin lru_cache para permitir override en tests).
    """
    backend = os.getenv("STORAGE_BACKEND", "local").lower()

    if backend == "minio":
        from app.services.storage_minio import MinIOStorageService
        return MinIOStorageService()
    elif backend == "s3":
        from app.services.storage_s3 import S3StorageService
        return S3StorageService()
    elif backend == "local":
        from app.services.storage_local import LocalStorageService
        return LocalStorageService()
    else:
        raise ValueError(
            f"STORAGE_BACKEND '{backend}' no reconocido. "
            "Valores válidos: local | minio | s3"
        )
