# Plan de Implementación — Fase 0: Observabilidad, Trazabilidad y Cost Tracking

> **Documento técnico de implementación**
> Basado en: [`roadmap.md`](../roadmap.md) · [`phase-0-observability.md`](phase-0-observability.md) · [`metrics-catalog.md`](metrics-catalog.md)
> **Decisiones tomadas:** Modelo de costes genérico/agnóstico · Migración única · 6 dashboards nuevos + enriquecer existentes · Tabla `resource_pricing` dinámica para todos los recursos

---

## Índice

1. [Arquitectura General](#1-arquitectura-general)
2. [0.1 — Schema DB, ResourcePricing y UsageCollector](#01--schema-db-resourcepricing-y-usagecollector)
3. [0.2 — Prometheus Custom Metrics](#02--prometheus-custom-metrics)
4. [0.3 — CostCalculator](#03--costcalculator)
5. [0.4 — API REST Endpoints](#04--api-rest-endpoints)
6. [0.5 — Grafana Dashboards](#05--grafana-dashboards)
7. [0.6 — Alert System + Celery Beat](#06--alert-system--celery-beat)
8. [0.7 — Budget, Forecast y Reportes](#07--budget-forecast-y-reportes)
9. [Secuencia de Ejecución](#9-secuencia-de-ejecución)
10. [Archivos — Resumen Completo](#10-archivos--resumen-completo)

---

## 1. Arquitectura General

```
backend/app/
├── observability/                          # NUEVO paquete
│   ├── __init__.py
│   ├── models.py                           # 5 ORM models (ResourcePricing incluido)
│   ├── metrics.py                          # 12 métricas Prometheus custom
│   ├── usage_collector.py                  # Singleton — captura + persiste
│   ├── cost_calculator.py                  # Singleton — precios dinámicos (DB+Redis)
│   ├── alert_engine.py                     # Evaluación y envío de alertas
│   └── reports.py                          # Generación PDF/CSV (0.7)
├── api/
│   ├── deps.py                             # MODIFICAR: + get_usage_collector, get_cost_calculator
│   └── routes/v1/
│       └── observability.py                # NUEVO router: 13 endpoints
├── schemas/
│   ├── observability.py                    # NUEVO: UsageQuery, CostQuery, AlertSchema, etc.
│   └── pricing.py                          # NUEVO: ResourcePricingCreate/Update/Response
├── models/
│   ├── __init__.py                         # MODIFICAR: + 5 imports nuevos modelos
│   └── tenant.py                           # MODIFICAR: + 4 columnas budget
├── worker/tasks/
│   └── scheduled_checks.py                 # NUEVO: drift checks, alert eval, zombie models
├── main.py                                 # MODIFICAR: registrar router + metrics
└── services/
    ├── inference_service.py                # MODIFICAR: hook post-inferencia
    └── training_service.py                 # MODIFICAR: hook post-entrenamiento
```

**Patrones a seguir** (basados en el codebase existente):

| Aspecto | Patrón establecido | Referencia |
|---------|-------------------|------------|
| ORM | `declarative_base` con `CustomBase` (camelCase→snake_case) | `models/base.py` |
| UUID PK | `Column(String, primary_key=True, default=generate_uuid)` | `models/tenant.py` |
| Timestamps | `created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))` | `models/tenant.py` |
| Servicios Singleton | Clase + variable global + función `get_*()` | `services/model_cache.py` |
| Dependencias FastAPI | `Depends(get_db)`, `Depends(get_current_user)` | `api/deps.py` |
| Rutas | `APIRouter(prefix=...)`, tags, dependencias por endpoint | `routes/v1/training.py` |
| Migraciones | naming `YYYYMMDD_HHMM_descripcion.py`, `upgrade()`/`downgrade()` | `migrations/versions/` |
| Métricas | `from prometheus_client import Counter, Gauge, Histogram` | `main.py` (`api_errors_total`) |
| Dashboards | JSON en `infra/grafana/provisioning/dashboards/`, registro en `dashboards.yml` | `infra/grafana/provisioning/` |

---

## 2. Sub-fase 0.1 — Schema DB, ResourcePricing y UsageCollector

**Esfuerzo:** 🟡 1-2 semanas · **Prioridad:** P0
**Dependencias:** Ninguna

### 2.1 Modelos ORM

5 nuevos modelos en `backend/app/observability/models.py`:

```python
from datetime import datetime, timezone
from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, String, JSON, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from app.models.base import Base
from app.utils.uuid import generate_uuid


class UsageMetric(Base):
    __tablename__ = "usage_metric"

    id = Column(String, primary_key=True, default=generate_uuid)
    tenant_id = Column(String, nullable=False, index=True)
    model_id = Column(String, nullable=True)
    domain = Column(String, nullable=False)         # tabular | nlp | cv | timeseries
    metric_name = Column(String, nullable=False)     # inference_latency_ms | training_duration_s | ...
    metric_value = Column(Float, nullable=False)
    tags = Column(JSON, nullable=True)               # {algorithm, task_type, gpu_type, ...}
    recorded_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        Index("idx_usage_tenant_time", "tenant_id", "recorded_at", postgresql_using="btree"),
        Index("idx_usage_model_metric", "model_id", "metric_name", "recorded_at"),
    )


class CostRecord(Base):
    __tablename__ = "cost_record"

    id = Column(String, primary_key=True, default=generate_uuid)
    tenant_id = Column(String, nullable=False)
    model_id = Column(String, nullable=True)
    cost_type = Column(String, nullable=False)      # compute | storage | api_tokens | inference | training
    amount = Column(Float, nullable=False)
    currency = Column(String, default="USD")
    source = Column(String, nullable=True)           # mlflow_run_id | prediction_id | manual_override
    period_start = Column(DateTime(timezone=True), nullable=False)
    period_end = Column(DateTime(timezone=True), nullable=False)
    metadata = Column(JSON, nullable=True)

    __table_args__ = (
        Index("idx_cost_tenant_period", "tenant_id", "period_start", postgresql_using="btree"),
    )


class DriftHistory(Base):
    __tablename__ = "drift_history"

    id = Column(String, primary_key=True, default=generate_uuid)
    model_id = Column(String, nullable=False)
    dataset_id = Column(String, nullable=True)
    tenant_id = Column(String, nullable=False)
    drift_type = Column(String, nullable=False)     # data_drift | concept_drift | model_drift
    metrics = Column(JSON, nullable=False)           # {psi: 0.15, ks: 0.03, drift_by_column: {...}}
    threshold_config = Column(JSON, nullable=True)
    is_anomaly = Column(Boolean, default=False)
    recorded_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        Index("idx_drift_model_time", "model_id", "recorded_at", postgresql_using="btree"),
    )


class AlertConfig(Base):
    __tablename__ = "alert_config"

    id = Column(String, primary_key=True, default=generate_uuid)
    tenant_id = Column(String, nullable=False)
    alert_type = Column(String, nullable=False)     # cost_spike | latency_degradation | drift_critical | ...
    metric_name = Column(String, nullable=False)
    threshold = Column(Float, nullable=False)
    comparison = Column(String, nullable=False)     # gt | lt | gte | lte | pct_change
    cooldown_min = Column(Integer, default=60)
    channels = Column(JSON, nullable=False)          # [{type: "webhook", url: "..."}, {type: "email", to: "..."}]
    enabled = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class ResourcePricing(Base):
    """
    Tabla de precios dinámica para todos los recursos:
    - compute_gpu (A100, V100, T4, L4, H100)
    - compute_cpu (standard)
    - storage (standard)
    - llm_tokens (gpt-4, claude-3-opus, etc.)
    """
    __tablename__ = "resource_pricing"

    id = Column(String, primary_key=True, default=generate_uuid)
    resource_category = Column(String, nullable=False)  # compute_gpu | compute_cpu | storage | llm_tokens
    resource_id = Column(String, nullable=False)         # A100 | gpt-4 | standard | ...
    pricing_unit = Column(String, nullable=False)        # per_hour | per_1k_input_tokens | per_gb_month
    price_per_unit = Column(Float, nullable=False)
    currency = Column(String, default="USD")
    metadata = Column(JSON, nullable=True)               # {source_url, context, notes}
    effective_from = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    effective_to = Column(DateTime(timezone=True), nullable=True)  # NULL = activo actualmente
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        Index("idx_resource_pricing_active", "resource_category", "resource_id", "pricing_unit",
              postgresql_where=text("effective_to IS NULL"), unique=True),
    )
```

### 2.2 Extensión del modelo Tenant

En `backend/app/models/tenant.py`, añadir:

```python
monthly_budget_usd = Column(Float, nullable=True)
budget_alert_pct = Column(Float, default=80.0)
billing_cycle = Column(String, default='monthly')    # monthly | prepaid
stripe_customer_id = Column(String, nullable=True)
```

### 2.3 Registro de modelos

En `backend/app/models/__init__.py`:

```python
from app.observability.models import (  # noqa: F401
    UsageMetric, CostRecord, DriftHistory, AlertConfig, ResourcePricing
)
```

### 2.4 Migración Alembic

Archivo: `backend/migrations/versions/20260721_1200_phase0_observability.py`

```
upgrade():
  op.create_table('usage_metric', ...)      ← 2 índices
  op.create_table('cost_record', ...)        ← 1 índice
  op.create_table('drift_history', ...)      ← 1 índice
  op.create_table('alert_config', ...)
  op.create_table('resource_pricing', ...)   ← partial unique index
  op.add_column('tenant', sa.Column('monthly_budget_usd', sa.Float()))
  op.add_column('tenant', sa.Column('budget_alert_pct', sa.Float(), server_default='80.0'))
  op.add_column('tenant', sa.Column('billing_cycle', sa.String(), server_default='monthly'))
  op.add_column('tenant', sa.Column('stripe_customer_id', sa.String()))

downgrade():
  op.drop_table('resource_pricing')
  op.drop_table('alert_config')
  op.drop_table('drift_history')
  op.drop_table('cost_record')
  op.drop_table('usage_metric')
  op.drop_column('tenant', 'monthly_budget_usd')
  op.drop_column('tenant', 'budget_alert_pct')
  op.drop_column('tenant', 'billing_cycle')
  op.drop_column('tenant', 'stripe_customer_id')
```

La migración también inserta los precios **seed**:

```python
from app.utils.uuid import generate_uuid

PRICING_SEED = [
    ('compute_gpu', 'A100', 'per_hour', 3.50),
    ('compute_gpu', 'V100', 'per_hour', 2.00),
    ('compute_gpu', 'T4', 'per_hour', 0.80),
    ('compute_gpu', 'L4', 'per_hour', 1.20),
    ('compute_cpu', 'standard', 'per_hour', 0.10),
    ('storage', 'standard', 'per_gb_month', 0.10),
    ('llm_tokens', 'gpt-4', 'per_1k_input_tokens', 0.03),
    ('llm_tokens', 'gpt-4', 'per_1k_output_tokens', 0.06),
    ('llm_tokens', 'gpt-4-turbo', 'per_1k_input_tokens', 0.01),
    ('llm_tokens', 'gpt-4-turbo', 'per_1k_output_tokens', 0.03),
    ('llm_tokens', 'gpt-3.5', 'per_1k_input_tokens', 0.0015),
    ('llm_tokens', 'gpt-3.5', 'per_1k_output_tokens', 0.002),
    ('llm_tokens', 'claude-3-opus', 'per_1k_input_tokens', 0.015),
    ('llm_tokens', 'claude-3-opus', 'per_1k_output_tokens', 0.075),
    ('llm_tokens', 'claude-3-sonnet', 'per_1k_input_tokens', 0.003),
    ('llm_tokens', 'claude-3-sonnet', 'per_1k_output_tokens', 0.015),
]
```

### 2.5 UsageCollector

`backend/app/observability/usage_collector.py`

```python
import logging
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy.orm import Session

from app.observability.metrics import (
    inference_latency_ms,
    predictions_total,
    training_duration_seconds,
    model_storage_bytes,
    ws_connections_active,
)

logger = logging.getLogger(__name__)


class UsageCollector:
    """
    Captura métricas de cada operación y las persiste en usage_metrics
    + expone a Prometheus.
    """

    def record_inference(
        self,
        db: Session,
        tenant_id: str,
        model_id: str,
        domain: str,
        latency_ms: float,
        tokens_input: int = 0,
        tokens_output: int = 0,
        success: bool = True,
        tags: Optional[dict] = None,
    ) -> None:
        # Persistir en usage_metrics
        from app.observability.models import UsageMetric

        metric = UsageMetric(
            tenant_id=tenant_id,
            model_id=model_id,
            domain=domain,
            metric_name="inference_latency_ms",
            metric_value=latency_ms,
            tags=tags or {},
        )
        db.add(metric)

        # Prometheus histogram
        algo = (tags or {}).get("algorithm", "unknown")
        inference_latency_ms.labels(
            tenant_id=tenant_id, domain=domain, algorithm=algo, success=str(success).lower()
        ).observe(latency_ms)

        predictions_total.labels(
            tenant_id=tenant_id, model_id=model_id, status="success" if success else "error"
        ).inc()

        if tokens_input or tokens_output:
            from app.observability import metrics as m
            m.inference_tokens_total.labels(
                tenant_id=tenant_id, model_id=model_id, direction="input"
            ).inc(tokens_input)
            m.inference_tokens_total.labels(
                tenant_id=tenant_id, model_id=model_id, direction="output"
            ).inc(tokens_output)

    def record_training(
        self,
        db: Session,
        tenant_id: str,
        model_id: str,
        domain: str,
        duration_s: float,
        gpu_hours: float = 0,
        cpu_hours: float = 0,
        metrics: Optional[dict] = None,
        tags: Optional[dict] = None,
    ) -> None:
        from app.observability.models import UsageMetric

        metric = UsageMetric(
            tenant_id=tenant_id,
            model_id=model_id,
            domain=domain,
            metric_name="training_duration_s",
            metric_value=duration_s,
            tags=tags or {},
        )
        db.add(metric)

        algo = (tags or {}).get("algorithm", "unknown")
        training_duration_seconds.labels(
            tenant_id=tenant_id, domain=domain, algorithm=algo
        ).observe(duration_s)

    def record_storage(
        self,
        tenant_id: str,
        model_id: str,
        bytes_count: float,
        storage_type: str = "model",
    ) -> None:
        model_storage_bytes.labels(
            tenant_id=tenant_id, model_id=model_id
        ).set(bytes_count)

    def record_ws_connection(self, tenant_id: str, delta: int = 1) -> None:
        """delta: +1 al conectar, -1 al desconectar"""
        ws_connections_active.labels(tenant_id=tenant_id).inc(delta)


# ── Singleton + FastAPI dependency ────────────────────────────────────────

_usage_collector: UsageCollector | None = None


def get_usage_collector() -> UsageCollector:
    global _usage_collector
    if _usage_collector is None:
        _usage_collector = UsageCollector()
    return _usage_collector
```

### 2.6 Dependencia FastAPI

En `backend/app/api/deps.py`:

```python
from app.observability.usage_collector import get_usage_collector, UsageCollector
from app.observability.cost_calculator import get_cost_calculator, CostCalculator
```

Nuevas dependencias:

```python
def get_usage_collector_dep() -> UsageCollector:
    return get_usage_collector()

def get_cost_calculator_dep() -> CostCalculator:
    return get_cost_calculator()
```

---

## 3. Sub-fase 0.2 — Prometheus Custom Metrics

**Esfuerzo:** 🟡 2 semanas · **Prioridad:** P0
**Dependencias:** 0.1 · `services/inference_service.py` · `services/training_service.py` · `routes/v1/streaming.py`

### 3.1 Módulo de métricas

`backend/app/observability/metrics.py` — 12 métricas:

| Métrica | Tipo | Labels | Buckets (si Histogram) |
|---------|------|--------|------------------------|
| `praxisml_inference_latency_ms` | Histogram | tenant_id, domain, algorithm, success | `[10,25,50,100,250,500,1000,2500,5000,10000]` |
| `praxisml_inference_tokens_total` | Counter | tenant_id, model_id, direction | — |
| `praxisml_inference_cost_usd` | Counter | tenant_id, model_id, cost_type | — |
| `praxisml_training_duration_seconds` | Histogram | tenant_id, domain, algorithm | `[10,30,60,120,300,600,1800,3600,7200]` |
| `praxisml_training_cost_usd` | Counter | tenant_id, domain, algorithm | — |
| `praxisml_model_storage_bytes` | Gauge | tenant_id, model_id | — |
| `praxisml_ws_connections_active` | Gauge | tenant_id | — |
| `praxisml_drift_score` | Gauge | model_id, metric | — |
| `praxisml_quota_usage_pct` | Gauge | tenant_id, resource | — |
| `praxisml_cost_budget_remaining` | Gauge | tenant_id | — |
| `praxisml_predictions_total` | Counter | tenant_id, model_id, status | — |
| `praxisml_errors_total` | Counter | tenant_id, error_type, component | — |

### 3.2 Puntos de integración

#### `InferenceService.predict()` (en `services/inference_service.py`)

Después de la inferencia exitosa, justo antes del `return`:

```python
from app.observability.usage_collector import get_usage_collector

# Al final del método predict()
collector = get_usage_collector()
collector.record_inference(
    db=db,  # inyectar db si está disponible, si no, postergar
    tenant_id=tenant_id,
    model_id=model.id,
    domain="tabular",
    latency_ms=elapsed_ms,
    success=True,
    tags={"algorithm": algorithm, "task_type": task_type},
)

# Si hay cost calculator disponible (0.3+), calcular y registrar coste
```

#### `SklearnTrainer.train()` y `PyTorchTrainer.train()` (en `services/training_service.py`)

Al finalizar el entrenamiento:

```python
from app.observability.usage_collector import get_usage_collector
from app.database import SessionLocal  # crear sesión propia (es Celery)

db = SessionLocal()
try:
    collector = get_usage_collector()
    collector.record_training(
        db=db,
        tenant_id=self.tenant_id,
        model_id=model.id,
        domain="tabular",
        duration_s=elapsed_s,
        cpu_hours=...,  # estimar desde duration_s
        gpu_hours=...,   # si aplica
        metrics={"f1": f1_score, "accuracy": acc},
        tags={"algorithm": algorithm, "task_type": task_type},
    )
    db.commit()
finally:
    db.close()
```

#### Conexiones WebSocket (en `routes/v1/streaming.py`)

```python
from app.observability.usage_collector import get_usage_collector

# En WebSocket.on_connect:
get_usage_collector().record_ws_connection(tenant_id, delta=+1)

# En WebSocket.on_disconnect:
get_usage_collector().record_ws_connection(tenant_id, delta=-1)
```

#### Cálculo de drift (en `routes/v1/drift.py`)

Después de cada cálculo de drift exitoso:

```python
from app.observability.metrics import drift_score

drift_score.labels(model_id=model_id, metric="psi").set(psi_value)
drift_score.labels(model_id=model_id, metric="ks").set(ks_value)
```

#### Quota checks (en `api/deps.py`)

Dentro de cada `check_*_quota()`:

```python
from app.observability.metrics import quota_usage_pct

pct = (current_count / limit) * 100
quota_usage_pct.labels(tenant_id=tenant.id, resource="datasets").set(pct)  # ej.
```

#### Exception handlers (en `main.py`)

Refactorizar los dos handlers existentes para usar el nuevo `errors_total` con labels:

```python
from app.observability.metrics import errors_total

errors_total.labels(
    tenant_id=getattr(request.state, "tenant_id", "unknown"),
    error_type="PraxisMLError",
    component=request.url.path,
).inc()
```

#### Señales Celery (en `worker/celery_app.py`)

```python
from app.observability.metrics import errors_total

@celery_app.task_failure_handler.connect
def task_failure_handler(sender=None, exception=None, **kwargs):
    task_name = sender.name if sender else "unknown"
    errors_total.labels(
        tenant_id="system",
        error_type=f"CeleryTaskFailure",
        component=task_name,
    ).inc()
```

---

## 4. Sub-fase 0.3 — CostCalculator

**Esfuerzo:** 🟡 1-2 semanas · **Prioridad:** P0
**Dependencias:** 0.1, 0.2

### 4.1 Módulo CostCalculator

`backend/app/observability/cost_calculator.py`

En lugar de `COST_TABLES` hardcodeado, consulta `resource_pricing` con cache Redis:

```python
import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional

from sqlalchemy.orm import Session

from app.core.redis import get_async_redis  # async redis client
from app.observability.models import ResourcePricing, CostRecord

logger = logging.getLogger(__name__)

REDIS_PRICING_TTL = 300  # 5 minutos


class CostCalculator:
    """
    Motor de estimación de costes con precios dinámicos desde DB + cache Redis.
    """

    def get_price(
        self, db: Session, category: str, resource_id: str, pricing_unit: str
    ) -> Optional[float]:
        """Obtiene el precio activo desde Redis cache → DB fallback."""
        cache_key = f"pricing:{category}:{resource_id}:{pricing_unit}"

        # 1. Intentar Redis
        try:
            redis = get_async_redis()
            cached = redis.get(cache_key)
            if cached:
                return float(cached)
        except Exception:
            pass  # Redis no disponible, seguir a DB

        # 2. DB query
        pricing = (
            db.query(ResourcePricing)
            .filter(
                ResourcePricing.resource_category == category,
                ResourcePricing.resource_id == resource_id,
                ResourcePricing.pricing_unit == pricing_unit,
                ResourcePricing.effective_to.is_(None),
            )
            .first()
        )
        if pricing is None:
            logger.warning("Precio no encontrado: %s/%s/%s", category, resource_id, pricing_unit)
            return None

        # 3. Cachear en Redis
        try:
            redis = get_async_redis()
            redis.setex(cache_key, REDIS_PRICING_TTL, str(pricing.price_per_unit))
        except Exception:
            pass

        return pricing.price_per_unit

    def estimate_inference_cost(
        self,
        db: Session,
        domain: str,
        latency_ms: float,
        tokens: Optional[dict] = None,
        gpu_type: Optional[str] = None,
    ) -> float:
        """Estima coste de una inferencia."""
        total = 0.0

        if domain == "llm" and tokens:
            # Coste por tokens
            input_price = self.get_price(db, "llm_tokens", tokens.get("model"), "per_1k_input_tokens")
            output_price = self.get_price(db, "llm_tokens", tokens.get("model"), "per_1k_output_tokens")
            if input_price:
                total += (tokens.get("input", 0) / 1000) * input_price
            if output_price:
                total += (tokens.get("output", 0) / 1000) * output_price
        elif domain in ("tabular", "nlp", "cv", "timeseries"):
            # Coste por tiempo de GPU/CPU
            hours = (latency_ms / 1000) / 3600
            if gpu_type:
                gpu_price = self.get_price(db, "compute_gpu", gpu_type, "per_hour")
                if gpu_price:
                    total += hours * gpu_price
            else:
                cpu_price = self.get_price(db, "compute_cpu", "standard", "per_hour")
                if cpu_price:
                    total += hours * cpu_price

        return total

    def estimate_training_cost(
        self,
        db: Session,
        gpu_hours: float = 0,
        gpu_type: Optional[str] = None,
        cpu_hours: float = 0,
    ) -> float:
        """Estima coste de un entrenamiento."""
        total = 0.0
        if gpu_hours > 0 and gpu_type:
            gpu_price = self.get_price(db, "compute_gpu", gpu_type, "per_hour")
            if gpu_price:
                total += gpu_hours * gpu_price
        if cpu_hours > 0:
            cpu_price = self.get_price(db, "compute_cpu", "standard", "per_hour")
            if cpu_price:
                total += cpu_hours * cpu_price
        return total

    def record_cost(
        self,
        db: Session,
        tenant_id: str,
        cost_type: str,
        amount: float,
        source: str,
        period_start: datetime,
        period_end: datetime,
        model_id: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> CostRecord:
        """Persiste un registro de coste y actualiza métricas Prometheus."""
        record = CostRecord(
            tenant_id=tenant_id,
            model_id=model_id,
            cost_type=cost_type,
            amount=amount,
            source=source,
            period_start=period_start,
            period_end=period_end,
            metadata=metadata or {},
        )
        db.add(record)

        # Actualizar métrica Prometheus
        from app.observability.metrics import inference_cost_usd, training_cost_usd

        if cost_type in ("inference", "api_tokens"):
            inference_cost_usd.labels(
                tenant_id=tenant_id, model_id=model_id or "unknown", cost_type=cost_type
            ).inc(amount)
        elif cost_type in ("training", "compute"):
            training_cost_usd.labels(
                tenant_id=tenant_id, domain=metadata.get("domain", "unknown") if metadata else "unknown",
                algorithm=metadata.get("algorithm", "unknown") if metadata else "unknown",
            ).inc(amount)

        # Actualizar budget remaining
        self._update_budget_gauge(db, tenant_id)
        return record

    def get_monthly_cost(self, db: Session, tenant_id: str) -> float:
        """Coste agregado del mes actual para un tenant."""
        now = datetime.now(timezone.utc)
        month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

        result = (
            db.query(func.coalesce(func.sum(CostRecord.amount), 0))
            .filter(
                CostRecord.tenant_id == tenant_id,
                CostRecord.period_start >= month_start,
            )
            .scalar()
        )
        return float(result)

    def budget_remaining(self, db: Session, tenant_id: str) -> Optional[float]:
        """Presupuesto restante del tenant."""
        from app.models.tenant import Tenant

        tenant = db.query(Tenant).filter(Tenant.id == tenant_id).first()
        if not tenant or tenant.monthly_budget_usd is None:
            return None
        spent = self.get_monthly_cost(db, tenant_id)
        return tenant.monthly_budget_usd - spent

    def _update_budget_gauge(self, db: Session, tenant_id: str) -> None:
        """Actualiza el gauge de presupuesto restante en Prometheus."""
        remaining = self.budget_remaining(db, tenant_id)
        if remaining is not None:
            from app.observability.metrics import cost_budget_remaining
            cost_budget_remaining.labels(tenant_id=tenant_id).set(max(remaining, 0))


# ── Singleton ─────────────────────────────────────────────────────────────

_cost_calculator: CostCalculator | None = None


def get_cost_calculator() -> CostCalculator:
    global _cost_calculator
    if _cost_calculator is None:
        _cost_calculator = CostCalculator()
    return _cost_calculator
```

### 4.2 Integración post-inferencia

En `InferenceService.predict()`, tras la inferencia:

```python
from app.observability.usage_collector import get_usage_collector
from app.observability.cost_calculator import get_cost_calculator

# Registrar métrica de uso
collector = get_usage_collector()
collector.record_inference(db=db, tenant_id=..., model_id=..., domain="tabular",
                           latency_ms=elapsed, success=True, tags=...)

# Calcular y registrar coste
calculator = get_cost_calculator()
cost = calculator.estimate_inference_cost(db, domain="tabular", latency_ms=elapsed)
if cost > 0:
    calculator.record_cost(db, tenant_id=..., cost_type="inference", amount=cost,
                           source=prediction_id, period_start=now, period_end=now)
```

### 4.3 Integración post-entrenamiento

En `SklearnTrainer.train()` y `PyTorchTrainer.train()`:

```python
from app.observability.usage_collector import get_usage_collector
from app.observability.cost_calculator import get_cost_calculator

# Registrar uso
collector = get_usage_collector()
collector.record_training(db=db, tenant_id=self.tenant_id, model_id=model.id,
                          domain="tabular", duration_s=elapsed, tags=...)

# Calcular y registrar coste
calculator = get_cost_calculator()
cost = calculator.estimate_training_cost(db, gpu_hours=gpu_h, gpu_type=gpu_type, cpu_hours=cpu_h)
if cost > 0:
    calculator.record_cost(db, tenant_id=self.tenant_id, cost_type="training", amount=cost,
                           source=mlflow_run_id, period_start=start, period_end=end,
                           metadata={"algorithm": algorithm, "domain": "tabular"})
```

---

## 5. Sub-fase 0.4 — API REST Endpoints

**Esfuerzo:** 🟡 2 semanas · **Prioridad:** P1
**Dependencias:** 0.1, 0.2, 0.3

### 5.1 Router

Archivo: `backend/app/api/routes/v1/observability.py`

```python
"""
Router de observabilidad: usage, costs, pricing, drift, alerts, health.
"""
import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.api.deps import (
    get_db,
    get_current_tenant,
    get_current_user,
    get_usage_collector_dep,
    get_cost_calculator_dep,
    require_admin,
    require_editor,
    require_viewer,
)
from app.models.tenant import Tenant
from app.models.user import User
from app.observability.usage_collector import UsageCollector
from app.observability.cost_calculator import CostCalculator
from app.schemas.observability import (
    UsageQueryParams, UsagePointResponse, UsageAggregationResponse,
    CostQueryParams, CostBreakdownResponse, CostForecastResponse,
    AlertCreate, AlertUpdate, AlertResponse,
    DriftPointResponse, HealthScoreResponse,
    MonthlyReportResponse,
)
from app.schemas.pricing import (
    ResourcePricingCreate, ResourcePricingResponse,
)

router = APIRouter(tags=["Observability"])
logger = logging.getLogger(__name__)


# ── Usage ─────────────────────────────────────────────────────────────────

@router.get("/usage")
def get_usage(
    params: UsageQueryParams = Depends(),
    tenant: Tenant = Depends(get_current_tenant),
    db: Session = Depends(get_db),
    _user: User = Depends(require_editor),
):
    """Métricas agregadas del tenant."""
    ...


@router.get("/usage/models/{model_id}")
def get_model_usage(
    model_id: str,
    tenant: Tenant = Depends(get_current_tenant),
    db: Session = Depends(get_db),
    _user: User = Depends(require_editor),
):
    """Métricas de un modelo específico."""
    ...


# ── Costs ─────────────────────────────────────────────────────────────────

@router.get("/costs")
def get_costs(
    params: CostQueryParams = Depends(),
    tenant: Tenant = Depends(get_current_tenant),
    db: Session = Depends(get_db),
    _user: User = Depends(require_editor),
):
    """Desglose de costes."""
    ...


@router.get("/costs/forecast")
def get_cost_forecast(
    tenant: Tenant = Depends(get_current_tenant),
    db: Session = Depends(get_db),
    calculator: CostCalculator = Depends(get_cost_calculator_dep),
    _user: User = Depends(require_editor),
):
    """Predicción de coste mensual."""
    ...


# ── Pricing (admin only) ──────────────────────────────────────────────────

@router.post("/pricing")
def create_pricing(
    data: ResourcePricingCreate,
    db: Session = Depends(get_db),
    _user: User = Depends(require_admin),
):
    """Crear/actualizar precio. Desactiva el anterior si existe."""
    # 1. Desactivar precio activo previo
    db.query(ResourcePricing).filter(
        ResourcePricing.resource_category == data.resource_category,
        ResourcePricing.resource_id == data.resource_id,
        ResourcePricing.pricing_unit == data.pricing_unit,
        ResourcePricing.effective_to.is_(None),
    ).update({"effective_to": datetime.now(timezone.utc)})

    # 2. Insertar nuevo
    pricing = ResourcePricing(**data.model_dump())
    db.add(pricing)
    db.commit()
    db.refresh(pricing)

    # 3. Invalidar cache Redis
    ...

    return pricing


@router.get("/pricing")
def list_pricing(
    category: str | None = Query(None),
    db: Session = Depends(get_db),
    _user: User = Depends(require_editor),
):
    """Listar precios activos."""
    query = db.query(ResourcePricing).filter(ResourcePricing.effective_to.is_(None))
    if category:
        query = query.filter(ResourcePricing.resource_category == category)
    return query.all()


@router.get("/pricing/history")
def get_pricing_history(
    category: str | None = Query(None),
    resource_id: str | None = Query(None),
    db: Session = Depends(get_db),
    _user: User = Depends(require_admin),
):
    """Historial completo de cambios de precio."""
    ...


# ── Drift ─────────────────────────────────────────────────────────────────

@router.get("/drift/history/{model_id}")
def get_drift_history(
    model_id: str,
    tenant: Tenant = Depends(get_current_tenant),
    db: Session = Depends(get_db),
    _user: User = Depends(require_editor),
):
    """Serie temporal de drift para un modelo."""
    ...


# ── Alerts ────────────────────────────────────────────────────────────────

@router.post("/alerts")
def create_alert(
    data: AlertCreate,
    tenant: Tenant = Depends(get_current_tenant),
    db: Session = Depends(get_db),
    _user: User = Depends(require_editor),
):
    """Crear alerta."""
    ...


@router.get("/alerts")
def list_alerts(
    tenant: Tenant = Depends(get_current_tenant),
    db: Session = Depends(get_db),
    _user: User = Depends(require_editor),
):
    """Listar alertas del tenant."""
    ...


@router.patch("/alerts/{alert_id}")
def update_alert(
    alert_id: str,
    data: AlertUpdate,
    tenant: Tenant = Depends(get_current_tenant),
    db: Session = Depends(get_db),
    _user: User = Depends(require_editor),
):
    """Actualizar alerta."""
    ...


@router.delete("/alerts/{alert_id}")
def delete_alert(
    alert_id: str,
    tenant: Tenant = Depends(get_current_tenant),
    db: Session = Depends(get_db),
    _user: User = Depends(require_admin),
):
    """Eliminar alerta."""
    ...


# ── Model Health ──────────────────────────────────────────────────────────

@router.get("/models/{model_id}/health")
def get_model_health(
    model_id: str,
    tenant: Tenant = Depends(get_current_tenant),
    db: Session = Depends(get_db),
    _user: User = Depends(require_viewer),
):
    """Health Score compuesto (0-100). Combina: accuracy, drift, latency, cost."""
    ...


# ── Report ────────────────────────────────────────────────────────────────

@router.get("/tenant/{tenant_id}/report")
def get_tenant_report(
    tenant_id: str,
    format: str = Query("pdf"),
    month: str | None = Query(None),
    db: Session = Depends(get_db),
    _user: User = Depends(require_admin),
):
    """Reporte mensual PDF/CSV."""
    ...
```

### 5.2 Registro en main.py

```python
from app.api.routes.v1 import observability

app.include_router(
    observability.router,
    prefix=f"{settings.API_V1_STR}/observability",
    tags=["Observability"],
)
```

### 5.3 Schemas Pydantic

`backend/app/schemas/observability.py`:

| Schema | Campos clave | Uso |
|--------|-------------|-----|
| `UsageQueryParams` | tenant_id?, from?, to?, domain?, metric?, group_by? | GET /usage |
| `UsagePointResponse` | recorded_at, metric_name, metric_value, tags | Response item |
| `UsageAggregationResponse` | metric, avg, p50, p95, p99, min, max, count | GET /usage response |
| `CostQueryParams` | tenant_id?, period?, cost_type?, from?, to? | GET /costs |
| `CostBreakdownResponse` | cost_type, total_amount, percentage, model_breakdown[] | GET /costs response |
| `CostForecastResponse` | projected_month_end_usd, budget_usd, overrun_risk_pct, daily_trend_usd | GET /costs/forecast |
| `DriftPointResponse` | recorded_at, drift_type, psi, ks, is_anomaly | GET /drift/history |
| `HealthScoreResponse` | score(0-100), components{accuracy, drift, latency, cost}, status | GET /models/{id}/health |
| `AlertCreate` | alert_type, metric_name, threshold, comparison, cooldown_min, channels | POST /alerts |
| `AlertUpdate` | threshold?, enabled?, channels?, ... | PATCH /alerts/{id} |
| `AlertResponse` | id + todos los campos de AlertConfig | Response |
| `MonthlyReportResponse` | url o contenido binario del reporte | GET /report |

`backend/app/schemas/pricing.py`:

| Schema | Campos clave | Uso |
|--------|-------------|-----|
| `ResourcePricingCreate` | resource_category, resource_id, pricing_unit, price_per_unit, currency, metadata | POST /pricing |
| `ResourcePricingResponse` | id + todos los campos | GET /pricing |

---

## 6. Sub-fase 0.5 — Grafana Dashboards

**Esfuerzo:** 🔴 2-3 semanas · **Prioridad:** P1
**Dependencias:** 0.2, 0.3

### 6.1 Nuevos dashboards

6 archivos JSON en `infra/grafana/provisioning/dashboards/`:

| Archivo | Título | Descripción | Paneles clave |
|---------|--------|-------------|---------------|
| `praxisml_cost_center.json` | Cost Center | Visión global de costes por tenant | Coste/día (bar chart), breakdown por modelo (pie), coste vs presupuesto (gauge), forecast mensual (line) |
| `praxisml_model_performance.json` | Model Performance | Métricas detalladas por modelo individual | Latencia p50/p95/p99 timeline, throughput, error rate, drift score histórico, uncertainty timeline |
| `praxisml_tenant_usage.json` | Tenant Usage | Uso global del tenant | Requests/día, entrenamientos/día, almacenamiento total, cuotas %, usuarios activos |
| `praxisml_llm_spend.json` | LLM Spend | Costes específicos de LLM | Tokens input/output/día, coste por modelo (bar), budget burn rate (line), coste por usuario |
| `praxisml_training_dashboard.json` | Training Dashboard | Métricas de entrenamiento | Duración por algoritmo, coste por entrenamiento, GPU utilization, success rate |
| `praxisml_drift_timeline.json` | Drift Timeline | Evolución temporal del drift | PSI/KS timeline por modelo, forecast line (regresión), alert annotations |

### 6.2 Dashboards existentes a enriquecer

| Dashboard | Paneles a añadir |
|-----------|-----------------|
| `praxisml_training.json` | Coste por entrenamiento (gauge), GPU utilization (gauge), training success rate (stat) |
| `praxisml_models.json` | Almacenamiento por modelo (bar chart), últimas drift scores (table), model health score (gauge) |
| `praxisml_api.json` | Latencia p95 por endpoint (bar), error rate por componente (pie), throughput global (stat) |
| `praxisml_tenants.json` | Budget remaining por tenant (gauge), quota usage (bar), billing cycle (table) |
| `praxisml_predictions.json` | Throughput por modelo (line), inference cost acumulado (area), predict success rate (stat) |

### 6.3 Registro

`infra/grafana/provisioning/dashboards/dashboards.yml` ya carga todos los JSON del directorio. Solo añadir los archivos.

---

## 7. Sub-fase 0.6 — Alert System + Celery Beat

**Esfuerzo:** 🔴 3 semanas · **Prioridad:** P1
**Dependencias:** 0.2, 0.3, 0.5

### 7.1 Celery Beat — nuevo servicio Docker

En `docker-compose.yml`:

```yaml
celery-beat:
  build:
    context: ./backend
    dockerfile: Dockerfile
  command: uv run celery -A app.worker.celery_app beat --loglevel=info
  env_file:
    - .env
    - .env.infra
  volumes:
    - ./backend:/app
  depends_on:
    redis:
      condition: service_started
    db:
      condition: service_healthy
```

### 7.2 Schedule en celery_app.py

En `backend/app/worker/celery_app.py`:

```python
from celery.schedules import crontab

celery_app.conf.beat_schedule = {
    "drift-check-every-6h": {
        "task": "app.worker.tasks.scheduled_checks.scheduled_drift_check",
        "schedule": crontab(minute=0, hour="*/6"),
    },
    "alert-evaluation-every-5m": {
        "task": "app.worker.tasks.scheduled_checks.evaluate_alerts",
        "schedule": crontab(minute="*/5"),
    },
    "zombie-model-check-daily": {
        "task": "app.worker.tasks.scheduled_checks.check_zombie_models",
        "schedule": crontab(hour=3, minute=0),
    },
    "budget-check-daily": {
        "task": "app.worker.tasks.scheduled_checks.check_budget_thresholds",
        "schedule": crontab(hour=9, minute=0),
    },
}
```

Añadir `"app.worker.tasks.scheduled_checks"` al `include` de Celery.

### 7.3 Tareas programadas

`backend/app/worker/tasks/scheduled_checks.py`:

```python
"""
Tareas periódicas para observabilidad.
Ejecutadas por Celery Beat según schedule en celery_app.py.
"""

import logging
from datetime import datetime, timezone, timedelta

from app.database import SessionLocal
from app.observability.models import DriftHistory, AlertConfig
from app.observability.metrics import drift_score
from app.observability.alert_engine import AlertEngine
from app.worker.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(name="app.worker.tasks.scheduled_checks.scheduled_drift_check")
def scheduled_drift_check():
    """
    Itera modelos en Production, calcula drift, persiste en drift_history,
    actualiza praxisml_drift_score gauge.
    """
    db = SessionLocal()
    try:
        from app.models.ml_model import MLModel

        production_models = (
            db.query(MLModel)
            .filter(MLModel.stage == "Production", MLModel.is_active == True)
            .all()
        )
        for model in production_models:
            try:
                # Reutilizar lógica de drift existente
                from app.api.routes.v1.drift import calculate_drift_for_model

                result = calculate_drift_for_model(model.id, db)
                if result:
                    # Persistir
                    history = DriftHistory(
                        model_id=model.id,
                        tenant_id=model.tenant_id,
                        drift_type="data_drift",
                        metrics={"psi": result["psi"], "ks": result["ks"]},
                        is_anomaly=result["psi"] > 0.2,
                    )
                    db.add(history)

                    # Actualizar gauge
                    drift_score.labels(model_id=model.id, metric="psi").set(result["psi"])
                    drift_score.labels(model_id=model.id, metric="ks").set(result["ks"])
                    db.commit()
            except Exception as e:
                logger.error("Error en drift check para modelo %s: %s", model.id, e)
                db.rollback()
    finally:
        db.close()


@celery_app.task(name="app.worker.tasks.scheduled_checks.evaluate_alerts")
def evaluate_alerts():
    """
    Evalúa todas las alert_config activas contra métricas actuales.
    Dispara notificaciones si el threshold se cruza y cooldown no está activo.
    """
    db = SessionLocal()
    try:
        alerts = db.query(AlertConfig).filter(AlertConfig.enabled == True).all()
        engine = AlertEngine()
        for alert in alerts:
            try:
                result = engine.evaluate(db, alert)
                if result.triggered:
                    engine.dispatch(alert, result)
                    engine.set_cooldown(alert)
            except Exception as e:
                logger.error("Error evaluando alerta %s: %s", alert.id, e)
    finally:
        db.close()


@celery_app.task(name="app.worker.tasks.scheduled_checks.check_zombie_models")
def check_zombie_models():
    """
    Detecta modelos en Production sin inferencias en los últimos 7 días.
    """
    db = SessionLocal()
    try:
        from app.models.ml_model import MLModel

        cutoff = datetime.now(timezone.utc) - timedelta(days=7)
        zombies = (
            db.query(MLModel)
            .filter(
                MLModel.stage == "Production",
                MLModel.is_active == True,
                MLModel.updated_at < cutoff,
            )
            .all()
        )
        for model in zombies:
            logger.warning("Modelo zombie detectado: %s (tenant=%s)", model.id, model.tenant_id)
            # Disparar alerta zombie
    finally:
        db.close()


@celery_app.task(name="app.worker.tasks.scheduled_checks.check_budget_thresholds")
def check_budget_thresholds():
    """
    Por cada tenant con budget, si coste mensual > budget_alert_pct%, dispara alerta.
    """
    from app.models.tenant import Tenant
    from app.observability.cost_calculator import get_cost_calculator

    db = SessionLocal()
    try:
        calculator = get_cost_calculator()
        tenants = db.query(Tenant).filter(Tenant.monthly_budget_usd.isnot(None)).all()
        for tenant in tenants:
            spent = calculator.get_monthly_cost(db, tenant.id)
            pct = (spent / tenant.monthly_budget_usd) * 100 if tenant.monthly_budget_usd else 0
            if pct >= (tenant.budget_alert_pct or 80):
                logger.warning(
                    "Budget alert: tenant=%s, spent=%.2f, budget=%.2f, pct=%.1f%%",
                    tenant.id, spent, tenant.monthly_budget_usd, pct,
                )
                # Disparar alerta budget_near_limit
    finally:
        db.close()
```

### 7.4 Alert Engine

`backend/app/observability/alert_engine.py`:

```python
"""
Motor de evaluación y despacho de alertas.
Soporta canales: webhook (Slack/Teams), email, dashboard badge, WS push.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional

import httpx
from sqlalchemy.orm import Session

from app.core.config import settings
from app.observability.models import AlertConfig

logger = logging.getLogger(__name__)


class AlertResult:
    def __init__(self, triggered: bool, current_value: float, threshold: float,
                 message: str, severity: str = "media"):
        self.triggered = triggered
        self.current_value = current_value
        self.threshold = threshold
        self.message = message
        self.severity = severity


class AlertEngine:
    """
    Evalúa alertas contra métricas y las despacha por los canales configurados.
    """

    def evaluate(self, db: Session, alert: AlertConfig) -> AlertResult:
        """Evalúa si una alerta debe dispararse."""
        current = self._get_current_metric(db, alert.metric_name, alert.tenant_id)
        if current is None:
            return AlertResult(triggered=False, current_value=0, threshold=alert.threshold, message="")

        triggered = self._compare(current, alert.threshold, alert.comparison)
        severity = self._determine_severity(alert.alert_type, triggered)

        return AlertResult(
            triggered=triggered,
            current_value=current,
            threshold=alert.threshold,
            message=self._build_message(alert, current),
            severity=severity,
        )

    def dispatch(self, alert: AlertConfig, result: AlertResult) -> None:
        """Envía la notificación por todos los canales configurados."""
        channels = alert.channels or []
        for channel in channels:
            try:
                if channel["type"] == "webhook":
                    self._send_webhook(channel["url"], alert, result)
                elif channel["type"] == "email":
                    self._send_email(channel["to"], alert, result)
                # WebSocket push se maneja via Redis pub/sub
                self._publish_ws(alert.tenant_id, alert, result)
            except Exception as e:
                logger.error("Error enviando alerta por %s: %s", channel["type"], e)

    def set_cooldown(self, alert: AlertConfig) -> None:
        """Marca cooldown en Redis para evitar spam."""
        from app.core.redis import get_async_redis
        try:
            redis = get_async_redis()
            key = f"alert_cooldown:{alert.id}"
            redis.setex(key, alert.cooldown_min * 60, "1")
        except Exception:
            pass

    def is_in_cooldown(self, alert: AlertConfig) -> bool:
        """Verifica si la alerta está en periodo de cooldown."""
        from app.core.redis import get_async_redis
        try:
            redis = get_async_redis()
            key = f"alert_cooldown:{alert.id}"
            return redis.exists(key) > 0
        except Exception:
            return False

    def _get_current_metric(self, db: Session, metric_name: str, tenant_id: str) -> Optional[float]:
        """Obtiene el valor actual de una métrica desde usage_metrics."""
        from app.observability.models import UsageMetric

        result = (
            db.query(UsageMetric.metric_value)
            .filter(
                UsageMetric.tenant_id == tenant_id,
                UsageMetric.metric_name == metric_name,
            )
            .order_by(UsageMetric.recorded_at.desc())
            .first()
        )
        return float(result[0]) if result else None

    def _compare(self, current: float, threshold: float, comparison: str) -> bool:
        comparators = {
            "gt": lambda c, t: c > t,
            "lt": lambda c, t: c < t,
            "gte": lambda c, t: c >= t,
            "lte": lambda c, t: c <= t,
        }
        return comparators.get(comparison, lambda c, t: False)(current, threshold)

    def _determine_severity(self, alert_type: str, triggered: bool) -> str:
        if not triggered:
            return "ok"
        critical_types = {"budget_exceeded", "latency_spike", "error_rate_spike", "drift_critical"}
        return "crítica" if alert_type in critical_types else "alta"

    def _build_message(self, alert: AlertConfig, current: float) -> str:
        return (
            f"[{alert.alert_type}] {alert.metric_name} = {current:.2f} "
            f"({alert.comparison} {alert.threshold})"
        )

    def _send_webhook(self, url: str, alert: AlertConfig, result: AlertResult) -> None:
        payload = {
            "text": result.message,
            "severity": result.severity,
            "tenant_id": alert.tenant_id,
            "alert_type": alert.alert_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        httpx.post(url, json=payload, timeout=5)

    def _send_email(self, to: str, alert: AlertConfig, result: AlertResult) -> None:
        """Implementar envío via SMTP o servicio de emails."""
        logger.info("Email alert a %s: %s", to, result.message)

    def _publish_ws(self, tenant_id: str, alert: AlertConfig, result: AlertResult) -> None:
        """Publica alerta en Redis pub/sub para push en tiempo real al frontend."""
        from app.core.redis import get_async_redis
        try:
            redis = get_async_redis()
            channel = f"praxisml:alerts:{tenant_id}"
            payload = json.dumps({
                "type": "alert",
                "alert_id": alert.id,
                "alert_type": alert.alert_type,
                "message": result.message,
                "severity": result.severity,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            redis.publish(channel, payload)
        except Exception:
            pass
```

### 7.5 15 alertas predefinidas

Seed que se insertan via script o endpoint admin al crear un tenant:

| Alerta | Métrica | Threshold | Comparación | Cooldown | Severidad |
|--------|---------|-----------|-------------|----------|-----------|
| cost_spike | inference_cost_usd | 2x promedio 7d rolling | pct_change | 60 min | Alta |
| budget_near_limit | cost_budget_remaining | 80% del budget | lt | 60 min | Media |
| budget_exceeded | cost_budget_remaining | 0 | lt | 1440 min | Crítica |
| latency_degradation | inference_latency_ms | 2x baseline semanal | gt | 30 min | Alta |
| latency_spike | inference_latency_ms | 5x baseline | gt | 15 min | Crítica |
| drift_warning | drift_score | threshold_config | gt | 120 min | Media |
| drift_critical | drift_score | threshold × 3 chequeos | gt | 60 min | Alta |
| token_spike_llm | inference_tokens_total | 3x promedio semanal | pct_change | 60 min | Alta |
| throughput_drop | predictions_total | 50% promedio horario | lt | 30 min | Media |
| error_rate_spike | errors_total | 5% en ventana 5 min | gt | 10 min | Crítica |
| training_failed | errors_total | >0 en ventana 1h | gt | 60 min | Media |
| quota_imminent | quota_usage_pct | 90% | gt | 1440 min | Baja |
| model_zombie | predictions_total | 0 en 7 días | lt | 1440 min | Baja |
| gpu_underutilized | training_gpu_utilization | 20% por >1h | lt | 120 min | Media |
| cache_inefficient | ... | ... | ... | ... | Media |

---

## 8. Sub-fase 0.7 — Budget, Forecast y Reportes

**Esfuerzo:** 🟢 1 semana · **Prioridad:** P2
**Dependencias:** 0.3, 0.6

### 8.1 Budget en Tenant API

Modificar `backend/app/api/routes/v1/tenants.py`:

- `PATCH /tenants/{id}` — aceptar `monthly_budget_usd`, `budget_alert_pct`, `billing_cycle`, `stripe_customer_id`
- `GET /tenants/{id}` — incluir los 4 campos en la respuesta

Actualizar `backend/app/schemas/tenant.py` con los campos nuevos.

### 8.2 Forecast de coste

Endpoint `GET /observability/costs/forecast`:

```python
from datetime import datetime, timezone, timedelta
from sqlalchemy import func

def forecast_monthly_cost(db: Session, tenant_id: str) -> CostForecastResponse:
    from app.observability.models import CostRecord

    # Últimos 90 días de costes diarios
    start = datetime.now(timezone.utc) - timedelta(days=90)
    daily = (
        db.query(
            func.date_trunc('day', CostRecord.period_start).label('day'),
            func.sum(CostRecord.amount).label('total'),
        )
        .filter(CostRecord.tenant_id == tenant_id, CostRecord.period_start >= start)
        .group_by(func.date_trunc('day', CostRecord.period_start))
        .order_by('day')
        .all()
    )

    if len(daily) < 7:
        return {"projected": None, "budget": None, "overrun_risk_pct": None, "daily_trend": None}

    # Regresión lineal simple
    x = list(range(len(daily)))
    y = [d.total for d in daily]
    n = len(x)
    slope = (n * sum(xi * yi for xi, yi in zip(x, y)) - sum(x) * sum(y)) / (n * sum(xi**2 for xi in x) - sum(x)**2)
    intercept = (sum(y) - slope * sum(x)) / n

    # Proyección a fin de mes
    days_in_month = (datetime.now(timezone.utc).replace(day=28) + timedelta(days=4)).day
    today = datetime.now(timezone.utc).day
    remaining_days = days_in_month - today + 1
    projected = intercept + slope * (n + remaining_days)

    # Budget
    from app.models.tenant import Tenant
    tenant = db.query(Tenant).filter(Tenant.id == tenant_id).first()
    budget = tenant.monthly_budget_usd

    return CostForecastResponse(
        projected_month_end_usd=round(projected, 2),
        budget_usd=budget,
        overrun_risk_pct=round((projected / budget) * 100, 1) if budget else None,
        daily_trend_usd=round(slope, 4),
    )
```

### 8.3 Reportes

`backend/app/observability/reports.py`:

```python
"""
Generación de reportes mensuales PDF y CSV.
"""

import csv
import io
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy.orm import Session

from app.observability.models import CostRecord, UsageMetric


def generate_cost_csv(db: Session, tenant_id: str, from_date: datetime, to_date: datetime) -> bytes:
    """Genera CSV con desglose de costes."""
    records = (
        db.query(CostRecord)
        .filter(
            CostRecord.tenant_id == tenant_id,
            CostRecord.period_start >= from_date,
            CostRecord.period_start <= to_date,
        )
        .order_by(CostRecord.period_start)
        .all()
    )

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["period_start", "period_end", "cost_type", "amount", "currency", "model_id", "source"])
    for r in records:
        writer.writerow([r.period_start.isoformat(), r.period_end.isoformat(),
                         r.cost_type, r.amount, r.currency, r.model_id, r.source])

    return output.getvalue().encode("utf-8")


def generate_monthly_report_pdf(db: Session, tenant_id: str, month: str) -> bytes:
    """Genera reporte PDF usando reportlab."""
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    # Parsear mes: "2026-07"
    year, month_num = month.split("-")
    from_date = datetime(int(year), int(month_num), 1, tzinfo=timezone.utc)
    if int(month_num) == 12:
        to_date = datetime(int(year) + 1, 1, 1, tzinfo=timezone.utc)
    else:
        to_date = datetime(int(year), int(month_num) + 1, 1, tzinfo=timezone.utc)

    # Consultar datos
    total_cost = (
        db.query(func.coalesce(func.sum(CostRecord.amount), 0))
        .filter(CostRecord.tenant_id == tenant_id, CostRecord.period_start >= from_date, CostRecord.period_start < to_date)
        .scalar()
    )

    buf = io.BytesIO()
    p = canvas.Canvas(buf, pagesize=A4)
    p.setTitle(f"PraxisML - Report Mensual {month}")
    p.drawString(72, 800, f"Reporte Mensual - Tenant {tenant_id}")
    p.drawString(72, 780, f"Periodo: {month}")
    p.drawString(72, 760, f"Coste Total: ${float(total_cost):.2f}")
    p.save()

    return buf.getvalue()
```

### 8.4 Nuevas dependencias

En `backend/pyproject.toml`:

```toml
"reportlab>=4.2,<5",
"httpx>=0.27",
```

### 8.5 Hooks post-inferencia y post-entrenamiento

**Archivos modificados:**

1. `backend/app/worker/tasks/predict.py` — Hook post-inferencia batch
2. `backend/app/worker/tasks/single_predict.py` — Hook post-inferencia individual
3. `backend/app/worker/tasks/train.py` — Hook post-entrenamiento
4. `backend/app/services/training_service.py` — Tracking de duración en `train()`

**Lógica del hook:**
```python
# En predict.py / single_predict.py (después de inference_time_s):
collector = get_usage_collector()
collector.record_inference(
    db=db, tenant_id=tenant_id, model_id=model_id,
    domain="tabular", latency_ms=inference_time_s * 1000,
    success=True, tags={"algorithm": ..., "method": method},
)

cost = calculator.estimate_inference_cost(db=db, domain="tabular", latency_ms=inference_time_s * 1000)
if cost > 0:
    calculator.record_cost(
        db=db, tenant_id=tenant_id, cost_type="inference", amount=cost,
        source=prediction_id, period_start=now, period_end=now, model_id=model_id,
    )
db.commit()

# En train.py (después de registrar modelo en BD):
collector = get_usage_collector()
collector.record_training(
    db=db, tenant_id=tenant_id, model_id=new_model.id,
    domain="tabular", duration_s=result.get("duration_s", 0),
    cpu_hours=result.get("duration_s", 0) / 3600,
    metrics=metrics, tags={"algorithm": algorithm, "task_type": task_type, "framework": framework},
)

cost = calculator.estimate_training_cost(
    db=db, cpu_hours=result.get("duration_s", 0) / 3600,
)
if cost > 0:
    calculator.record_cost(
        db=db, tenant_id=tenant_id, cost_type="training", amount=cost,
        source=mlflow_run_id, period_start=now, period_end=now,
        model_id=new_model.id, metadata={"algorithm": algorithm, "domain": "tabular"},
    )
db.commit()

# En training_service.py:
# Añadir tracking de duración en train():
t_start = time.perf_counter()
result = self._train_holdout(...)  # o self._train_cv(...)
result["duration_s"] = time.perf_counter() - t_start
return result
```

---

## 9. Secuencia de Ejecución

```
Semana 1-2   ████████████████░░░░░░░░░░░░░░░░ 0.1  Schema DB + ResourcePricing + UsageCollector + migration
Semana 3-4   ████████████████████████████████ 0.2  12 Prometheus metrics + 6 integraciones + Celery signals
Semana 5-6   ████████████████████████████████ 0.3  CostCalculator (DB+Redis cache) + hooks entrenamiento
Semana 7-8   ████████████████░░░░░░░░░░░░░░░░ 0.4  13 API endpoints + schemas Pydantic
Semana 9-11  ████████████████████████████████ 0.5  6 nuevos dashboards + enriquecer 5 existentes
Semana 12-14 ████████████████████████████████ 0.6  Celery Beat + 4 tareas + alert engine + 15 alertas
Semana 15    ████████░░░░░░░░░░░░░░░░░░░░░░░░ 0.7  Budget API fields + forecast + reportes PDF/CSV
```

Las sub-fases **0.4** y **0.5** pueden ejecutarse en paralelo si hay 2 personas.

---

## 10. Archivos — Resumen Completo

### Nuevos (19)

| # | Archivo | Sub-fase |
|---|---------|----------|
| 1 | `backend/app/observability/__init__.py` | 0.1 |
| 2 | `backend/app/observability/models.py` | 0.1 |
| 3 | `backend/app/observability/metrics.py` | 0.2 |
| 4 | `backend/app/observability/usage_collector.py` | 0.1 |
| 5 | `backend/app/observability/cost_calculator.py` | 0.3 |
| 6 | `backend/app/observability/alert_engine.py` | 0.6 |
| 7 | `backend/app/observability/reports.py` | 0.7 |
| 8 | `backend/app/api/routes/v1/observability.py` | 0.4 |
| 9 | `backend/app/schemas/observability.py` | 0.4 |
| 10 | `backend/app/schemas/pricing.py` | 0.4 |
| 11 | `backend/app/worker/tasks/scheduled_checks.py` | 0.6 |
| 12 | `infra/grafana/provisioning/dashboards/praxisml_cost_center.json` | 0.5 |
| 13 | `infra/grafana/provisioning/dashboards/praxisml_model_performance.json` | 0.5 |
| 14 | `infra/grafana/provisioning/dashboards/praxisml_tenant_usage.json` | 0.5 |
| 15 | `infra/grafana/provisioning/dashboards/praxisml_llm_spend.json` | 0.5 |
| 16 | `infra/grafana/provisioning/dashboards/praxisml_training_dashboard.json` | 0.5 |
| 17 | `infra/grafana/provisioning/dashboards/praxisml_drift_timeline.json` | 0.5 |
| 18 | `backend/migrations/versions/20260721_1200_phase0_observability.py` | 0.1 |
| 19 | `roadmap/phase-0-implementation-plan.md` | — |

### Modificados (11)

| # | Archivo | Cambio | Sub-fase |
|---|---------|--------|----------|
| 1 | `backend/app/models/tenant.py` | +4 columnas budget | 0.1 |
| 2 | `backend/app/models/__init__.py` | +5 imports de observability.models | 0.1 |
| 3 | `backend/app/api/deps.py` | +2 dependencias (usage_collector, cost_calculator) | 0.1 |
| 4 | `backend/app/main.py` | +registrar router observability + refactor exception handlers | 0.1/0.2 |
| 5 | `backend/app/services/inference_service.py` | +hook post-inferencia (usage_collector + cost_calculator) | 0.2/0.3 |
| 6 | `backend/app/services/training_service.py` | +hook post-entrenamiento | 0.2/0.3 |
| 7 | `backend/app/worker/celery_app.py` | +beat_schedule + task signals | 0.2/0.6 |
| 8 | `backend/app/api/routes/v1/drift.py` | +actualizar gauge drift_score | 0.2 |
| 9 | `backend/app/api/routes/v1/tabular/streaming.py` | +record_ws_connection en on_connect/disconnect | 0.2 |
| 10 | `backend/app/api/routes/v1/tenants.py` | +campos budget en PATCH/GET | 0.7 |
| 11 | `docker-compose.yml` | +servicio celery-beat | 0.6 |
| 12 | `backend/pyproject.toml` | +reportlab, httpx | 0.7 |

---

> ⚠️ **Antes de empezar:** Asegúrate de que Redis esté configurado con `MAXMEMORY_POLICY allkeys-lru` y suficiente memoria para el caché de precios + cooldowns de alertas.
>
> ✅ **Verificación post-implementación:** `GET /metrics` debe exponer las 12 métricas nuevas, `GET /api/v1/observability/pricing` debe devolver los 16 precios seed, y `GET /health` debe seguir respondiendo `{"status": "ok"}`.
