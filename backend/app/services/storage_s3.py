"""
Implementación S3StorageService — backend AWS S3.

Idéntica interfaz que MinIOStorageService pero apuntando
a los endpoints de AWS S3 (sin endpoint_url personalizado).

Variables de entorno requeridas:
    AWS_ACCESS_KEY_ID       — IAM access key
    AWS_SECRET_ACCESS_KEY   — IAM secret access key
    AWS_DEFAULT_REGION      — Región de AWS (p.ej. eu-west-1)
    S3_BUCKET               — Nombre del bucket S3
"""

from __future__ import annotations

import os
from typing import BinaryIO

import boto3
from botocore.exceptions import ClientError

from app.core.exceptions import StorageError, StorageObjectNotFoundError
from app.core.logging import get_logger
from app.services.storage_service import StorageService

logger = get_logger(__name__)


class S3StorageService(StorageService):

    def __init__(self):
        self.bucket = os.getenv("S3_BUCKET", "praxisml-prod")
        region = os.getenv("AWS_DEFAULT_REGION", "eu-west-1")

        # boto3 lee AWS_ACCESS_KEY_ID y AWS_SECRET_ACCESS_KEY automáticamente
        self._client = boto3.client("s3", region_name=region)

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
            logger.info("Objeto subido a S3", extra={"detail": f"s3://{self.bucket}/{key}"})
            return key
        except ClientError as exc:
            raise StorageError(f"Error subiendo {key} a S3", detail=str(exc)) from exc

    def download(self, key: str) -> bytes:
        try:
            response = self._client.get_object(Bucket=self.bucket, Key=key)
            return response["Body"].read()
        except ClientError as exc:
            error_code = exc.response["Error"]["Code"]
            if error_code in ("NoSuchKey", "404"):
                raise StorageObjectNotFoundError(key)
            raise StorageError(f"Error descargando {key} desde S3", detail=str(exc)) from exc

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
            logger.warning("Error eliminando objeto de S3", extra={"detail": str(exc)})

    def exists(self, key: str) -> bool:
        try:
            self._client.head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError:
            return False
