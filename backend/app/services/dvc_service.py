"""
DVC Service for data versioning.
Provides centralized management of datasets with DVC.
"""

import hashlib
import logging
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.core.config import settings

logger = logging.getLogger(__name__)


class DVCService:
    def __init__(self, tenant_id: str):
        self.tenant_id = tenant_id
        # Alineamos con LocalStorageService: Path(settings.DATA_DIR) / "storage"
        self.storage_base = Path(settings.DATA_DIR) / "storage"
        self.tenant_dir = self.storage_base / "tenants" / tenant_id
        self.data_dir = self.tenant_dir / "datasets"
        self.dvc_dir = self.tenant_dir # El root del repo DVC es el root del tenant en el storage
        self.tenant_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def get_local_path(self, storage_key: str) -> Path:
        """Convierte una storage key en una ruta local absoluta para DVC."""
        return self.storage_base / storage_key

    def _run_command(
        self, cmd: List[str], cwd: Optional[Path] = None
    ) -> subprocess.CompletedProcess:
        """Run a shell command and return the result."""
        working_dir = cwd or self.dvc_dir
        try:
            result = subprocess.run(
                cmd,
                cwd=str(working_dir),
                capture_output=True,
                text=True,
                check=True,
            )
            return result
        except subprocess.CalledProcessError as e:
            logger.error(f"Command failed: {' '.join(cmd)}")
            logger.error(f"Error: {e.stderr}")
            raise e

    def init_repository(self) -> bool:
        """Initialize DVC repository for the tenant."""
        try:
            if not (self.dvc_dir / ".dvc").exists():
                self._run_command(["dvc", "init", "--no-scm"])
            return True
        except Exception as e:
            logger.error(f"Failed to init DVC repository: {e}")
            return False

    def configure_remote(self, remote_name: str = "minio") -> bool:
        """Configure the DVC remote (MinIO/S3)."""
        try:
            # Check if remote already exists
            remotes = self._run_command(["dvc", "remote", "list"]).stdout
            if remote_name not in remotes:
                self._run_command(
                    [
                        "dvc",
                        "remote",
                        "add",
                        "-d",
                        remote_name,
                        f"s3://praxisml-dvc/{self.tenant_id}",
                    ]
                )

            # Use the environment variables or settings for endpoint
            endpoint = os.getenv("AWS_ENDPOINT_URL_S3") or settings.MINIO_ENDPOINT
            # We must ensure DVC knows about the endpoint
            self._run_command(["dvc", "remote", "modify", remote_name, "endpointurl", endpoint])

            # Optional: ensure credentials if not picking up from ENV
            # self._run_command(["dvc", "remote", "modify", remote_name, "access_key_id", settings.MINIO_ACCESS_KEY])
            # self._run_command(["dvc", "remote", "modify", remote_name, "secret_access_key", settings.MINIO_SECRET_KEY])

            return True
        except Exception as e:
            logger.error(f"Failed to configure remote: {e}")
            return False

    def get_file_hash(self, file_path: str) -> str:
        """Calculate MD5 hash of a file."""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def add_dataset(self, file_path: str, registry_name: str) -> Dict[str, Any]:
        """
        Add a dataset to DVC tracking.
        Returns dict with hash, version info.
        """
        # Ensure DVC is initialized
        self.init_repository()

        # Get file hash before DVC tracking
        file_hash = self.get_file_hash(file_path)

        # Add to DVC
        relative_path = Path(file_path).relative_to(self.dvc_dir)
        self._run_command(["dvc", "add", str(relative_path)])

        # Get the .dvc file to extract info
        dvc_file = f"{file_path}.dvc"
        dvc_info = self._parse_dvc_file(dvc_file)

        # Push to remote
        try:
            self._run_command(["dvc", "push"])
        except Exception as e:
            logger.warning(f"Failed to push to remote: {e}")

        return {
            "hash": file_hash,
            "dvc_hash": dvc_info.get("md5", file_hash),
            "size": os.path.getsize(str(file_path)),
        }

    def _parse_dvc_file(self, dvc_file_path: str) -> Dict[str, Any]:
        """Parse a .dvc file to extract hash information."""
        if not os.path.exists(dvc_file_path):
            return {}

        try:
            with open(dvc_file_path, "r") as f:
                content = f.read()
                # Simple parsing - look for md5 hash
                if "md5:" in content:
                    for line in content.split("\n"):
                        if "md5:" in line:
                            return {"md5": line.split("md5:")[1].strip()}
        except Exception as e:
            logger.error(f"Failed to parse DVC file: {e}")

        return {}

    def get_version_history(self, registry_name: str) -> List[Dict[str, Any]]:
        """
        Get version history for a registry.
        This reads from the .dvc files in the datasets directory.
        """
        versions = []
        datasets_path = self.data_dir

        if not datasets_path.exists():
            return versions

        for dvc_file in datasets_path.glob("*.dvc"):
            try:
                # Parse the .dvc file
                info = self._parse_dvc_file(str(dvc_file))
                if info:
                    versions.append(
                        {
                            "file": dvc_file.stem,
                            "hash": info.get("md5"),
                        }
                    )
            except Exception as e:
                logger.warning(f"Failed to read {dvc_file}: {e}")

        return versions

    def pull_dataset(self, file_path: str) -> bool:
        """Pull a dataset from the DVC remote."""
        try:
            relative_path = Path(file_path).relative_to(self.dvc_dir)
            self._run_command(["dvc", "pull", str(relative_path)])
            return True
        except Exception as e:
            logger.error(f"Failed to pull dataset: {e}")
            return False

    def push_dataset(self, file_path: str) -> bool:
        """Push a dataset to the DVC remote."""
        try:
            relative_path = Path(file_path).relative_to(self.dvc_dir)
            self._run_command(["dvc", "push", str(relative_path)])
            return True
        except Exception as e:
            logger.error(f"Failed to push dataset: {e}")
            return False

    def remove_tracking(self, file_path: str) -> bool:
        """Remove a dataset from DVC tracking."""
        try:
            # Remove .dvc file
            dvc_file = f"{file_path}.dvc"
            if os.path.exists(dvc_file):
                os.remove(dvc_file)

            # Remove .gitignore entry
            gitignore_path = f"{os.path.dirname(file_path)}/.gitignore"
            if os.path.exists(gitignore_path):
                with open(gitignore_path, "r") as f:
                    lines = f.readlines()
                filename = os.path.basename(file_path)
                with open(gitignore_path, "w") as f:
                    f.writelines([line for line in lines if filename not in line])

            return True
        except Exception as e:
            logger.error(f"Failed to remove tracking: {e}")
            return False


def get_dvc_service(tenant_id: str) -> DVCService:
    """Factory function to get DVC service instance."""
    return DVCService(tenant_id)


def track_dataset_with_dvc(
    tenant_id: str, file_path: str, registry_name: str
) -> Dict[str, Any]:
    """
    Convenience function to track a dataset with DVC.
    Returns tracking information.
    """
    service = DVCService(tenant_id)

    # Initialize if needed
    service.init_repository()
    service.configure_remote()

    # Add to DVC
    result = service.add_dataset(file_path, registry_name)

    return {
        "is_dvc_tracked": True,
        "dvc_remote": "minio",
        "dvc_hash": result.get("hash"),
        "dvc_registry_name": registry_name,
        "dvc_version": 1,
        **result,
    }
