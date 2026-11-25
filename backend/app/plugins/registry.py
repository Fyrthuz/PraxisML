"""
PluginRegistry — Descubrimiento, registro e introspección de Plugins de Dominio.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import FastAPI

from app.plugins.contracts import DomainPlugin

logger = logging.getLogger(__name__)


class PluginRegistry:
    """Registro central de plugins de dominio activos en la instancia."""

    def __init__(self) -> None:
        self._plugins: Dict[str, DomainPlugin] = {}

    def register(self, plugin: DomainPlugin) -> None:
        """Registra un plugin por su nombre de dominio."""
        if plugin.name in self._plugins:
            logger.debug("Sobrescribiendo plugin registrado: %s", plugin.name)
        self._plugins[plugin.name] = plugin
        logger.info(
            "Plugin registrado: %s v%s (queue=%s)",
            plugin.name,
            plugin.version,
            plugin.default_queue,
        )

    def get(self, name: str) -> DomainPlugin:
        """Obtiene un plugin por nombre o lanza KeyError si no está activo."""
        if name not in self._plugins:
            raise KeyError(
                f"Plugin '{name}' no registrado. Disponibles: {list(self._plugins.keys())}"
            )
        return self._plugins[name]

    def has(self, name: str) -> bool:
        return name in self._plugins

    def list_plugins(self) -> List[DomainPlugin]:
        return list(self._plugins.values())

    def list_metadata(self) -> List[Dict[str, Any]]:
        return [p.to_metadata() for p in self._plugins.values()]

    def clear(self) -> None:
        self._plugins.clear()

    def register_routes(self, app: FastAPI, api_prefix: str) -> None:
        """Monta los routers de todos los plugins activos sobre la app FastAPI."""
        for plugin in self._plugins.values():
            for mount in plugin.get_routers():
                full_prefix = f"{api_prefix}{mount.prefix}"
                app.include_router(
                    mount.router,
                    prefix=full_prefix,
                    tags=mount.tags,
                    include_in_schema=mount.include_in_schema,
                )

    def register_copilot_tools(self, tool_registry: Any) -> int:
        """Registra los CopilotTool declarados por todos los plugins activos."""
        count = 0
        for plugin in self._plugins.values():
            for tool in plugin.get_copilot_tools():
                tool_registry.register(tool)
                count += 1
        return count

    def discover_builtin_plugins(
        self, enabled: Optional[set[str]] = None
    ) -> List[str]:
        """
        Carga plugins incorporados si están habilitados (`ENABLED_PLUGINS`) y
        sus dependencias opcionales están disponibles.
        Permite ejecutar el Control Plane solo o contenedores independientes por dominio.
        """
        from app.core.config import settings

        allowed = enabled if enabled is not None else settings.enabled_plugins_set
        loaded: List[str] = []
        candidates = [
            ("tabular", "app.plugins.tabular.plugin", "TabularPlugin"),
            ("nlp", "app.plugins.nlp.plugin", "NLPPlugin"),
        ]
        for name, module_path, cls_name in candidates:
            if allowed is not None and name not in allowed:
                logger.info("Plugin '%s' deshabilitado por ENABLED_PLUGINS.", name)
                continue
            if self.has(name):
                loaded.append(name)
                continue
            try:
                module = __import__(module_path, fromlist=[cls_name])
                plugin_cls = getattr(module, cls_name)
                self.register(plugin_cls())
                loaded.append(name)
            except ImportError as exc:
                logger.warning(
                    "Plugin opcional '%s' omitido (dependencias no instaladas): %s",
                    name,
                    exc,
                )
        return loaded


_registry: Optional[PluginRegistry] = None


def get_plugin_registry() -> PluginRegistry:
    """Devuelve la instancia singleton de PluginRegistry."""
    global _registry
    if _registry is None:
        _registry = PluginRegistry()
    return _registry
