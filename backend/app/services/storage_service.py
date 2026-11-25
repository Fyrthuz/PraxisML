"""
StorageService — Interfaz abstracta multi-cloud de almacenamiento de objetos
y registro pluggable de backends (StorageBackendRegistry).

Backends registrados por defecto (con carga lazy para evitar acoplar SDKs cloud):
    - "local"  -> disco local
    - "minio"  -> MinIO self-hosted (S3-compatible)
    - "s3"     -> AWS S3 / LocalStack
    - Extensible a "gcs", "azure" o cualquier adaptador personalizado mediante
      `register_storage_backend(name, factory)`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import BinaryIO, Callable, Dict, List, Optional

from app.core.config import settings


class StorageService(ABC):
    """Contrato agnóstico al proveedor cloud para almacenamiento de artefactos."""

    @abstractmethod
    def upload(
        self,
        key: str,
        data: bytes | BinaryIO,
        content_type: str = "application/octet-stream",
    ) -> str:
        """Sube dato/fichero al backend y retorna su object key."""
        ...

    @abstractmethod
    def download(self, key: str) -> bytes:
        """Descarga el objeto como bytes. Lanza StorageObjectNotFoundError si no existe."""
        ...

    @abstractmethod
    def get_url(self, key: str, expires_in: int = 3600) -> str:
        """Retorna una URL pre-firmada o referencia de acceso al objeto."""
        ...

    @abstractmethod
    def delete(self, key: str) -> None:
        """Elimina el objeto (idempotente si no existe)."""
        ...

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Comprueba si un objeto existe."""
        ...

    def list_objects(self, prefix: str = "") -> List[str]:
        """Lista las object keys bajo un prefijo dado."""
        raise NotImplementedError(
            f"{self.__class__.__name__} no implementa list_objects()"
        )

    def get_uri(self, key: str) -> str:
        """Devuelve la URI canónica del objeto (e.g. file://, s3://, gs://, abfs://)."""
        return key


class StorageBackendRegistry:
    """Registro pluggable de adaptadores de almacenamiento con carga lazy."""

    def __init__(self) -> None:
        self._factories: Dict[str, Callable[[], StorageService]] = {}
        self._register_builtin_adapters()

    def _register_builtin_adapters(self) -> None:
        def _local() -> StorageService:
            from app.services.storage_local import LocalStorageService

            return LocalStorageService()

        def _minio() -> StorageService:
            from app.services.storage_minio import MinIOStorageService

            return MinIOStorageService()

        def _s3() -> StorageService:
            from app.services.storage_s3 import S3StorageService

            return S3StorageService()

        self._factories["local"] = _local
        self._factories["minio"] = _minio
        self._factories["s3"] = _s3

    def register(self, name: str, factory: Callable[[], StorageService]) -> None:
        """Registra un adaptador de almacenamiento (e.g. gcs, azure, obstore)."""
        self._factories[name.lower()] = factory

    def list_backends(self) -> List[str]:
        return sorted(self._factories.keys())

    def create(self, name: str) -> StorageService:
        key = name.lower()
        factory = self._factories.get(key)
        if factory is None:
            available = " | ".join(self.list_backends())
            raise ValueError(
                f"STORAGE_BACKEND '{name}' no reconocido. Registrados: {available}"
            )
        return factory()


_storage_registry = StorageBackendRegistry()


def register_storage_backend(
    name: str, factory: Callable[[], StorageService]
) -> None:
    """Permite registrar nuevos proveedores cloud en tiempo de arranque o desde plugins."""
    _storage_registry.register(name, factory)


def get_storage_registry() -> StorageBackendRegistry:
    return _storage_registry


def get_storage(backend_name: Optional[str] = None) -> StorageService:
    """Instancia el StorageService configurado por STORAGE_BACKEND (o el nombre indicado)."""
    backend = (backend_name or settings.STORAGE_BACKEND).lower()
    return _storage_registry.create(backend)
