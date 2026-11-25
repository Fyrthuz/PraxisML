"""
ExperimentTracker & TrackerRegistry — Abstracción de tracking y registro de modelos.

Usa MLflow como backend por defecto y permite enchufar otros trackers
(Weights & Biases, ClearML, Vertex AI Experiments) mediante
`register_tracker_backend(name, factory)`.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional


class ExperimentTracker(ABC):
    """Contrato común de experiment tracking y model registry."""

    backend_name: str

    @abstractmethod
    def get_tracking_uri(self) -> str:
        ...

    @abstractmethod
    def load_model(self, run_id: str, device: Any = None) -> Any:
        ...

    @abstractmethod
    def get_run_details(self, run_id: str) -> Dict[str, Any]:
        ...

    @abstractmethod
    def list_registered_models(self) -> List[Dict[str, Any]]:
        ...


class MLflowTrackerBackend(ExperimentTracker):
    backend_name = "mlflow"

    def __init__(self) -> None:
        from app.platform.tracker.mlflow_service import MLFlowService

        self._svc = MLFlowService()

    def get_tracking_uri(self) -> str:
        return self._svc.get_tracking_uri()

    def load_model(self, run_id: str, device: Any = None) -> Any:
        import torch

        dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return self._svc.load_model(run_id, device=dev)

    def get_run_details(self, run_id: str) -> Dict[str, Any]:
        return self._svc.get_run_details(run_id)

    def list_registered_models(self) -> List[Dict[str, Any]]:
        return self._svc.list_registered_models()


class TrackerRegistry:
    """Registro pluggable de backends de experiment tracking."""

    def __init__(self) -> None:
        self._factories: Dict[str, Callable[[], ExperimentTracker]] = {
            "mlflow": MLflowTrackerBackend,
        }

    def register(self, name: str, factory: Callable[[], ExperimentTracker]) -> None:
        self._factories[name.lower()] = factory

    def list_backends(self) -> List[str]:
        return sorted(self._factories.keys())

    def create(self, name: str) -> ExperimentTracker:
        key = name.lower()
        factory = self._factories.get(key)
        if factory is None:
            raise ValueError(
                f"ExperimentTracker '{name}' no registrado (disponibles: {', '.join(self.list_backends())})."
            )
        return factory()


_tracker_registry = TrackerRegistry()


def register_tracker_backend(
    name: str, factory: Callable[[], ExperimentTracker]
) -> None:
    _tracker_registry.register(name, factory)


def get_tracker_registry() -> TrackerRegistry:
    return _tracker_registry


def get_experiment_tracker(backend_name: Optional[str] = None) -> ExperimentTracker:
    selected = (backend_name or os.getenv("EXPERIMENT_TRACKER", "mlflow")).lower()
    return _tracker_registry.create(selected)
