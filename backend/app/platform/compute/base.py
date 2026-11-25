"""
ComputeBackend — Contrato del Compute Plane y registro pluggable de backends
y ejecutores de entornos (Runtime Environments).

Permite despachar jobs por perfil de hardware y entorno de dependencias
(`EnvironmentSpec`) sobre Celery/uv en local y escalar a Ray, Docker o
Kubernetes Jobs sin modificar routers ni plugins.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from app.core.config import settings


class ComputeProfile(str, Enum):
    CPU_LIGHT = "cpu-light"
    CPU_HEAVY = "cpu-heavy"
    GPU_TRAIN = "gpu-train"
    SYSTEM_BEAT = "system-beat"


@dataclass
class EnvironmentSpec:
    """Especificación declarativa de un entorno aislado con dependencias propias."""

    name: str = "default"
    python_version: str = "3.10"
    base_image: str = "python:3.10-slim"
    packages: List[str] = field(default_factory=list)
    env_vars: Dict[str, str] = field(default_factory=dict)
    runner_type: str = "uv_isolated"  # uv_isolated | docker | k8s | ray


@dataclass
class JobSpec:
    """Especificación declarativa de un job a ejecutar en el Compute Plane."""

    task_name: str
    kwargs: Dict[str, Any] = field(default_factory=dict)
    args: tuple = ()
    profile: ComputeProfile = ComputeProfile.CPU_HEAVY
    tenant_id: Optional[str] = None
    environment: Optional[EnvironmentSpec] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class JobHandle:
    """Referencia opaca al job despachado."""

    job_id: str
    backend: str
    queue: str


@dataclass
class JobStatus:
    """Estado normalizado de un job en ejecución o finalizado."""

    job_id: str
    state: str  # PENDING | STARTED | SUCCESS | FAILURE | REVOKED
    result: Optional[Any] = None
    error: Optional[str] = None


class ComputeBackend(ABC):
    """Interfaz abstracta para backends de ejecución distribuida."""

    backend_name: str

    @abstractmethod
    def submit(self, spec: JobSpec) -> JobHandle:
        """Envía un job al perfil de compute solicitado."""
        ...

    @abstractmethod
    def get_status(self, job_id: str) -> JobStatus:
        """Consulta el estado actual de un job."""
        ...

    @abstractmethod
    def cancel(self, job_id: str) -> bool:
        """Solicita la cancelación de un job."""
        ...


class ComputeBackendRegistry:
    """Registro pluggable de backends de cómputo (celery, ray, k8s, etc.)."""

    def __init__(self) -> None:
        self._factories: Dict[str, Callable[[], ComputeBackend]] = {}
        self._register_builtins()

    def _register_builtins(self) -> None:
        def _celery() -> ComputeBackend:
            from app.platform.compute.celery_backend import CeleryComputeBackend

            return CeleryComputeBackend()

        self._factories["celery"] = _celery

    def register(self, name: str, factory: Callable[[], ComputeBackend]) -> None:
        self._factories[name.lower()] = factory

    def list_backends(self) -> List[str]:
        return sorted(self._factories.keys())

    def create(self, name: str) -> ComputeBackend:
        key = name.lower()
        factory = self._factories.get(key)
        if factory is None:
            available = ", ".join(self.list_backends())
            raise NotImplementedError(
                f"ComputeBackend '{name}' no registrado (disponibles: {available})."
            )
        return factory()


_compute_registry = ComputeBackendRegistry()


def register_compute_backend(
    name: str, factory: Callable[[], ComputeBackend]
) -> None:
    _compute_registry.register(name, factory)


def get_compute_registry() -> ComputeBackendRegistry:
    return _compute_registry


def get_compute_backend(backend_name: Optional[str] = None) -> ComputeBackend:
    """Factory del ComputeBackend según COMPUTE_BACKEND."""
    backend = (backend_name or settings.COMPUTE_BACKEND).lower()
    return _compute_registry.create(backend)
