"""
Implementación MinIOStorageService — backend MinIO self-hosted.

Usa el SDK boto3 apuntando al endpoint de MinIO.
Es 100 % compatible con la API de S3, por lo que el código
puede migrar a AWS S3 solo cambiando STORAGE_BACKEND=s3.

Variables de entorno requeridas:
    MINIO_ENDPOINT      — p.ej. http://minio:9000
    MINIO_ACCESS_KEY    — Access key de MinIO
    MINIO_SECRET_KEY    — Secret key de MinIO
    MINIO_BUCKET        — Nombre del bucket (default: antigravity)
    MINIO_REGION        — Región (default: us-east-1, requerido por boto3)
"""

from __future__ import annotations

import os
from typing import BinaryIO

import boto3
from botocore.exceptions import ClientError

from app.services.storage_service import StorageService
from app.core.exceptions import StorageError, StorageObjectNotFoundError
from app.core.logging import get_logger

logger = get_logger(__name__)


class MinIOStorageService(StorageService):

    def __init__(self):
        endpoint = os.getenv("MINIO_ENDPOINT", "http://minio:9000")
        access_key = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
        secret_key = os.getenv("MINIO_SECRET_KEY", "minioadmin")
        self.bucket = os.getenv("MINIO_BUCKET", "antigravity")
        region = os.getenv("MINIO_REGION", "us-east-1")

        self._client = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region,
        )
        self._ensure_bucket()

    def _ensure_bucket(self) -> None:
        """Crea el bucket si no existe."""
        try:
            self._client.head_bucket(Bucket=self.bucket)
        except ClientError as exc:
            error_code = exc.response["Error"]["Code"]
            if error_code in ("404", "NoSuchBucket"):
                try:
                    self._client.create_bucket(Bucket=self.bucket)
                    logger.info("Bucket creado en MinIO", extra={"detail": self.bucket})
                except ClientError as create_exc:
                    raise StorageError(
                        f"No se pudo crear el bucket '{self.bucket}'",
                        detail=str(create_exc),
                    ) from create_exc
            else:
                raise StorageError(
                    f"Error conectando a MinIO bucket '{self.bucket}'",
                    detail=str(exc),
                ) from exc

    def upload(self, key: str, data: bytes | BinaryIO, content_type: str = "application/octet-stream") -> str:
        try:
            if isinstance(data, bytes):
                import io
                data = io.BytesIO(data)
            self._client.upload_fileobj(
                data,
                self.bucket,
                key,
                ExtraArgs={"ContentType": content_type},
            )
            logger.info("Objeto subido a MinIO", extra={"detail": f"s3://{self.bucket}/{key}"})
            return key
        except ClientError as exc:
            raise StorageError(f"Error subiendo {key} a MinIO", detail=str(exc)) from exc

    def download(self, key: str) -> bytes:
        try:
            response = self._client.get_object(Bucket=self.bucket, Key=key)
            return response["Body"].read()
        except ClientError as exc:
            error_code = exc.response["Error"]["Code"]
            if error_code in ("NoSuchKey", "404"):
                raise StorageObjectNotFoundError(key)
            raise StorageError(f"Error descargando {key} desde MinIO", detail=str(exc)) from exc

    def get_url(self, key: str, expires_in: int = 3600) -> str:
        try:
            url = self._client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.bucket, "Key": key},
                ExpiresIn=expires_in,
            )
            return url
        except ClientError as exc:
            raise StorageError(f"Error generando URL pre-firmada para {key}", detail=str(exc)) from exc

    def delete(self, key: str) -> None:
        try:
            self._client.delete_object(Bucket=self.bucket, Key=key)
        except ClientError as exc:
            logger.warning("Error eliminando objeto de MinIO", extra={"detail": str(exc)})

    def exists(self, key: str) -> bool:
        try:
            self._client.head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError:
            return False
