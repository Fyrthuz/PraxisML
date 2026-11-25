"""
Contratos base para Plugins de Dominio ML.

Cada dominio (tabular, nlp, cv, timeseries) implementa `DomainPlugin` para
registrar sus rutas HTTP/WS, entrenador, motor de inferencia y tools del Copilot.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from fastapi import APIRouter


@dataclass
class RouterMount:
    """Describe cómo montar un APIRouter de un plugin en FastAPI."""

    router: APIRouter
    prefix: str
    tags: List[str]
    include_in_schema: bool = True


@dataclass
class TrainingContext:
    """Contexto unificado de entrenamiento independiente del dominio."""

    tenant_id: str
    dataset_ref: str
    algorithm: str
    task_type: str
    hyperparams: Dict[str, Any] = field(default_factory=dict)
    validation_config: Dict[str, Any] = field(default_factory=dict)
    workspace_id: Optional[str] = None
    compute_profile: str = "cpu-heavy"
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingResult:
    """Resultado normalizado devuelto por cualquier BaseTrainer."""

    mlflow_run_id: str
    metrics: Dict[str, float]
    artifact_keys: Dict[str, str] = field(default_factory=dict)
    duration_s: float = 0.0
    framework: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseTrainer(ABC):
    """Interfaz común para entrenadores de cualquier dominio ML."""

    domain: str

    @abstractmethod
    def supported_algorithms(self) -> Dict[str, Dict[str, Any]]:
        """Devuelve el catálogo de algoritmos soportados por este dominio."""
        ...

    @abstractmethod
    def train(self, ctx: TrainingContext) -> TrainingResult:
        """Ejecuta el entrenamiento según el TrainingContext recibido."""
        ...


class BaseInferenceEngine(ABC):
    """Interfaz común para motores de inferencia de cualquier dominio ML."""

    domain: str

    @abstractmethod
    def load(self, model_record: Any, storage: Optional[Any] = None) -> Any:
        """Carga el modelo en memoria (con caché y soporte de storage remoto)."""
        ...

    @abstractmethod
    def predict(self, loaded_model: Any, payload: Any, **kwargs) -> Dict[str, Any]:
        """Ejecuta inferencia sobre el payload y devuelve resultado estructurado."""
        ...


class DomainPlugin(ABC):
    """Contrato principal de un Plugin de Dominio en PraxisML."""

    name: str
    version: str = "1.0.0"
    required_extra: str = ""
    default_queue: str = "cpu-heavy"

    @abstractmethod
    def get_routers(self) -> List[RouterMount]:
        """Devuelve los routers HTTP/WS del dominio a montar en FastAPI."""
        ...

    @abstractmethod
    def get_trainer(self) -> BaseTrainer:
        """Devuelve el entrenador del dominio."""
        ...

    @abstractmethod
    def get_inference_engine(self) -> BaseInferenceEngine:
        """Devuelve el motor de inferencia del dominio."""
        ...

    def get_profiler(self) -> Optional[Any]:
        """Opcional: profiler de datasets específico del dominio."""
        return None

    def get_copilot_tools(self) -> List[Any]:
        """Opcional: tools tipados que este dominio registra en CopilotToolRegistry."""
        return []

    def register_event_handlers(self, bus: Any) -> None:
        """Opcional: suscripciones al Event Bus interno."""
        return None

    def to_metadata(self) -> Dict[str, Any]:
        """Serializa metadatos del plugin para introspección/API."""
        return {
            "name": self.name,
            "version": self.version,
            "required_extra": self.required_extra,
            "default_queue": self.default_queue,
            "algorithms_count": len(self.get_trainer().supported_algorithms()),
            "copilot_tools_count": len(self.get_copilot_tools()),
        }
