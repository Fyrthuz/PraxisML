"""
Implementación LocalStorageService — backend de disco local.

Útil para desarrollo y entornos sin MinIO/S3.
Las rutas de los objetos se mapean directamente a DATA_DIR/<key>.

Variables de entorno (opcionales):
    DATA_DIR  — directorio raíz (por defecto: backend/data)
"""

from __future__ import annotations

from pathlib import Path
from typing import BinaryIO

from app.services.storage_service import StorageService
from app.core.exceptions import StorageError, StorageObjectNotFoundError
from app.core.config import settings


class LocalStorageService(StorageService):

    def __init__(self):
        self.base_dir = Path(settings.DATA_DIR) / "storage"
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _resolve(self, key: str) -> Path:
        # Evita path traversal
        resolved = (self.base_dir / key).resolve()
        if not str(resolved).startswith(str(self.base_dir.resolve())):
            raise StorageError(f"Clave de almacenamiento no válida: {key}")
        return resolved

    def upload(self, key: str, data: bytes | BinaryIO, content_type: str = "application/octet-stream") -> str:
        try:
            path = self._resolve(key)
            path.parent.mkdir(parents=True, exist_ok=True)
            mode = "wb"
            if isinstance(data, bytes):
                path.write_bytes(data)
            else:
                with open(path, mode) as f:
                    f.write(data.read())
            return key
        except StorageError:
            raise
        except Exception as exc:
            raise StorageError(f"Error subiendo {key} a disco local", detail=str(exc)) from exc

    def download(self, key: str) -> bytes:
        path = self._resolve(key)
        if not path.exists():
            raise StorageObjectNotFoundError(key)
        try:
            return path.read_bytes()
        except Exception as exc:
            raise StorageError(f"Error descargando {key} desde disco local", detail=str(exc)) from exc

    def get_url(self, key: str, expires_in: int = 3600) -> str:
        # En local no hay URL real; devolvemos la ruta absoluta como referencia
        path = self._resolve(key)
        return f"file://{path}"

    def delete(self, key: str) -> None:
        path = self._resolve(key)
        if path.exists():
            path.unlink()

    def exists(self, key: str) -> bool:
        return self._resolve(key).exists()
