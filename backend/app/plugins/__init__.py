"""Sistema modular de Plugins de Dominio de PraxisML."""

from app.plugins.contracts import (
    BaseInferenceEngine,
    BaseTrainer,
    DomainPlugin,
    RouterMount,
    TrainingContext,
    TrainingResult,
)
from app.plugins.registry import PluginRegistry, get_plugin_registry

__all__ = [
    "BaseInferenceEngine",
    "BaseTrainer",
    "DomainPlugin",
    "PluginRegistry",
    "RouterMount",
    "TrainingContext",
    "TrainingResult",
    "get_plugin_registry",
]
