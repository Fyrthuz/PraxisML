"""
Event Bus interno desacoplado por backend (EventBusBackend + EventBusRegistry).
Soporta Redis Streams por defecto, buffer en memoria para tests aislados y
extensión pluggable a AWS SQS/SNS, Kafka o NATS mediante `register_event_bus_backend`.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional
from uuid import uuid4

from app.core.redis import create_sync_redis_client

logger = logging.getLogger(__name__)

STREAM_NAME = "praxisml:events"
DLQ_STREAM_NAME = "praxisml:events:dlq"


@dataclass
class DomainEvent:
    event_type: str
    tenant_id: str
    payload: Dict[str, Any] = field(default_factory=dict)
    event_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    source: str = "control-plane"

    def to_stream_fields(self) -> Dict[str, str]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "tenant_id": self.tenant_id,
            "source": self.source,
            "timestamp": self.timestamp,
            "payload": json.dumps(self.payload, default=str),
        }

    @classmethod
    def from_stream_fields(cls, fields: Dict[str, Any]) -> "DomainEvent":
        raw_payload = fields.get("payload", "{}")
        try:
            payload = json.loads(raw_payload) if isinstance(raw_payload, str) else raw_payload
        except Exception:
            payload = {"raw": str(raw_payload)}
        return cls(
            event_id=str(fields.get("event_id", uuid4())),
            event_type=str(fields.get("event_type", "unknown")),
            tenant_id=str(fields.get("tenant_id", "")),
            source=str(fields.get("source", "control-plane")),
            timestamp=str(fields.get("timestamp", datetime.now(timezone.utc).isoformat())),
            payload=payload if isinstance(payload, dict) else {"value": payload},
        )


EventHandler = Callable[[DomainEvent], None]


class EventBusBackend(ABC):
    """Contrato abstracto de transporte de eventos (Redis Streams, SQS, Kafka, NATS)."""

    backend_name: str

    @abstractmethod
    def emit(self, event: DomainEvent) -> None:
        ...

    @abstractmethod
    def emit_dlq(self, event: DomainEvent, error_message: str) -> None:
        ...

    @abstractmethod
    def fetch_recent(
        self, limit: int = 50, tenant_id: Optional[str] = None
    ) -> List[DomainEvent]:
        ...


class InMemoryEventBusBackend(EventBusBackend):
    backend_name = "memory"

    def __init__(self, max_size: int = 500) -> None:
        self._max_size = max_size
        self._events: List[DomainEvent] = []
        self._dlq: List[Dict[str, Any]] = []

    def emit(self, event: DomainEvent) -> None:
        self._events.append(event)
        if len(self._events) > self._max_size:
            self._events = self._events[-self._max_size :]

    def emit_dlq(self, event: DomainEvent, error_message: str) -> None:
        self._dlq.append({"event": event, "error": error_message[:500]})

    def fetch_recent(
        self, limit: int = 50, tenant_id: Optional[str] = None
    ) -> List[DomainEvent]:
        filtered = [
            e for e in reversed(self._events)
            if tenant_id is None or e.tenant_id == tenant_id
        ]
        return filtered[:limit]


class RedisStreamsEventBusBackend(EventBusBackend):
    backend_name = "redis"

    def __init__(
        self,
        stream_name: str = STREAM_NAME,
        dlq_stream: str = DLQ_STREAM_NAME,
    ) -> None:
        self.stream_name = stream_name
        self.dlq_stream = dlq_stream
        self._fallback = InMemoryEventBusBackend()

    def emit(self, event: DomainEvent) -> None:
        self._fallback.emit(event)
        client = create_sync_redis_client()
        if client is None:
            return
        try:
            client.xadd(
                self.stream_name,
                event.to_stream_fields(),
                maxlen=10_000,
                approximate=True,
            )
        except Exception as exc:
            logger.warning("Redis Streams XADD falló para %s: %s", event.event_type, exc)

    def emit_dlq(self, event: DomainEvent, error_message: str) -> None:
        self._fallback.emit_dlq(event, error_message)
        client = create_sync_redis_client()
        if client is None:
            return
        try:
            fields = event.to_stream_fields()
            fields["error"] = error_message[:500]
            client.xadd(self.dlq_stream, fields, maxlen=1_000, approximate=True)
        except Exception as exc:
            logger.debug("No se pudo escribir en DLQ: %s", exc)

    def fetch_recent(
        self, limit: int = 50, tenant_id: Optional[str] = None
    ) -> List[DomainEvent]:
        client = create_sync_redis_client()
        events: List[DomainEvent] = []
        if client is not None:
            try:
                raw_entries = client.xrevrange(self.stream_name, count=limit * 2)
                for _stream_id, fields in raw_entries:
                    ev = DomainEvent.from_stream_fields(fields)
                    if tenant_id is None or ev.tenant_id == tenant_id:
                        events.append(ev)
                    if len(events) >= limit:
                        break
            except Exception as exc:
                logger.debug("Fallback a buffer local para recent_events: %s", exc)

        if not events:
            return self._fallback.fetch_recent(limit=limit, tenant_id=tenant_id)
        return events


class EventBusRegistry:
    """Registro pluggable de transportes del Event Bus."""

    def __init__(self) -> None:
        self._factories: Dict[str, Callable[[], EventBusBackend]] = {
            "redis": RedisStreamsEventBusBackend,
            "memory": InMemoryEventBusBackend,
        }

    def register(self, name: str, factory: Callable[[], EventBusBackend]) -> None:
        self._factories[name.lower()] = factory

    def list_backends(self) -> List[str]:
        return sorted(self._factories.keys())

    def create(self, name: str) -> EventBusBackend:
        key = name.lower()
        factory = self._factories.get(key)
        if factory is None:
            raise ValueError(
                f"EventBusBackend '{name}' no registrado (disponibles: {', '.join(self.list_backends())})."
            )
        return factory()


_event_bus_registry = EventBusRegistry()


def register_event_bus_backend(
    name: str, factory: Callable[[], EventBusBackend]
) -> None:
    _event_bus_registry.register(name, factory)


class DomainEventBus:
    """Fachada de publicación/suscripción de eventos delegada al EventBusBackend activo."""

    def __init__(
        self,
        backend: Optional[EventBusBackend] = None,
        stream_name: str = STREAM_NAME,
        dlq_stream: str = DLQ_STREAM_NAME,
    ) -> None:
        self.backend = backend or RedisStreamsEventBusBackend(
            stream_name=stream_name, dlq_stream=dlq_stream
        )
        self._handlers: Dict[str, List[EventHandler]] = {}

    def subscribe(self, event_type: str, handler: EventHandler) -> None:
        self._handlers.setdefault(event_type, []).append(handler)

    def publish(
        self,
        event_type: str,
        tenant_id: str,
        payload: Optional[Dict[str, Any]] = None,
        source: str = "control-plane",
    ) -> DomainEvent:
        event = DomainEvent(
            event_type=event_type,
            tenant_id=tenant_id,
            payload=payload or {},
            source=source,
        )
        self.backend.emit(event)
        self._dispatch_local(event)
        return event

    def _dispatch_local(self, event: DomainEvent) -> None:
        handlers = self._handlers.get(event.event_type, []) + self._handlers.get("*", [])
        for handler in handlers:
            try:
                handler(event)
            except Exception as exc:
                logger.error("Error en handler de evento '%s': %s", event.event_type, exc)
                self.backend.emit_dlq(event, str(exc))

    def recent_events(
        self, limit: int = 50, tenant_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        events = self.backend.fetch_recent(limit=limit, tenant_id=tenant_id)
        return [
            {
                "event_id": e.event_id,
                "event_type": e.event_type,
                "tenant_id": e.tenant_id,
                "source": e.source,
                "timestamp": e.timestamp,
                "payload": e.payload,
            }
            for e in events
        ]


_event_bus: Optional[DomainEventBus] = None


def get_event_bus() -> DomainEventBus:
    global _event_bus
    if _event_bus is None:
        _event_bus = DomainEventBus()
    return _event_bus
