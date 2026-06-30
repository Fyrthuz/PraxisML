# Fase 0 — Observabilidad, Trazabilidad y Cost Tracking

**Carácter:** Transversal. Se construye incrementalmente en paralelo a las Fases 1-4.
**Duración estimada:** Continuo (sub-fases 0.1→0.7 a lo largo de Q1-Q2 2027)
**Dependencias:** Ninguna (0.1 es independiente); 0.2 requiere integración con training/inference services.

---

## Índice

- [Motivación](#motivación)
- [Modelo de Datos](#modelo-de-datos)
- [Métricas por Tipo de Modelo](#métricas-trackeables-por-tipo-de-modelo)
- [Componentes del Sistema](#componentes-del-sistema)
- [API Endpoints](#api-endpoints)
- [Alertas](#alertas-configurables)
- [Plan de Implementación](#plan-de-implementación)
- [Coste y ROI Estimado](#coste-y-roi-estimado)

---

## Motivación

Sin un sistema unificado de observabilidad:

- **No sabes cuánto cuesta cada modelo** → imposible justificar ROI
- **No sabes qué modelos están en producción realmente** → modelos zombies
- **No detectas drift hasta que el usuario se queja** → reactivo, no proactivo
- **No tienes histórico de métricas** → no puedes hacer forecasting
- **No puedes facturar a tus tenants por uso real** → modelo de negocio limitado

---

## Modelo de Datos

### Tabla: `usage_metrics`

Trazabilidad de uso por modelo y tenant — registro atómico de cada evento.

```sql
CREATE TABLE usage_metrics (
    id              UUID PRIMARY KEY,
    tenant_id       VARCHAR NOT NULL,
    model_id        VARCHAR,
    domain          VARCHAR NOT NULL,      -- tabular | nlp | cv | timeseries
    metric_name     VARCHAR NOT NULL,       -- inference_latency_ms | training_duration_s | ...
    metric_value    DOUBLE PRECISION NOT NULL,
    tags            JSONB,                  -- {algorithm, task_type, gpu_type, ...}
    recorded_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_usage_tenant_time ON usage_metrics(tenant_id, recorded_at DESC);
CREATE INDEX idx_usage_model ON usage_metrics(model_id, metric_name, recorded_at DESC);
```

### Tabla: `cost_records`

Costes agregados por tenant/recurso — cada registro representa un periodo facturable.

```sql
CREATE TABLE cost_records (
    id              UUID PRIMARY KEY,
    tenant_id       VARCHAR NOT NULL,
    model_id        VARCHAR,
    cost_type       VARCHAR NOT NULL,       -- compute | storage | api_tokens | inference | training
    amount          DOUBLE PRECISION NOT NULL,
    currency        VARCHAR DEFAULT 'USD',
    source          VARCHAR,                -- mlflow_run_id | prediction_id | manual_override
    period_start    TIMESTAMPTZ NOT NULL,
    period_end      TIMESTAMPTZ NOT NULL,
    metadata        JSONB
);

CREATE INDEX idx_cost_tenant_period ON cost_records(tenant_id, period_start DESC);
```

### Tabla: `drift_history`

Historial de drift persistido para análisis de evolución temporal.

```sql
CREATE TABLE drift_history (
    id              UUID PRIMARY KEY,
    model_id        VARCHAR NOT NULL,
    dataset_id      VARCHAR,
    tenant_id       VARCHAR NOT NULL,
    drift_type      VARCHAR NOT NULL,       -- data_drift | concept_drift | model_drift
    metrics         JSONB NOT NULL,          -- {psi: 0.15, ks: 0.03, drift_by_column: {...}}
    threshold_config JSONB,
    is_anomaly      BOOLEAN DEFAULT FALSE,
    recorded_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_drift_model_time ON drift_history(model_id, recorded_at DESC);
```

### Tabla: `alert_config`

Configuración de alertas por tenant.

```sql
CREATE TABLE alert_config (
    id              UUID PRIMARY KEY,
    tenant_id       VARCHAR NOT NULL,
    alert_type      VARCHAR NOT NULL,       -- cost_spike | latency_degradation | drift_critical | ...
    metric_name     VARCHAR NOT NULL,
    threshold       DOUBLE PRECISION NOT NULL,
    comparison      VARCHAR NOT NULL,       -- gt | lt | gte | lte | pct_change
    cooldown_min    INT DEFAULT 60,
    channels        JSONB NOT NULL,          -- [{type: "webhook", url: "..."}, {type: "email", to: "...}]
    enabled         BOOLEAN DEFAULT TRUE,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);
```

### Extensión del modelo `Tenant`

```sql
ALTER TABLE tenant ADD COLUMN monthly_budget_usd DOUBLE PRECISION;
ALTER TABLE tenant ADD COLUMN budget_alert_pct DOUBLE PRECISION DEFAULT 80.0;
ALTER TABLE tenant ADD COLUMN billing_cycle VARCHAR DEFAULT 'monthly';  -- monthly | prepaid
ALTER TABLE tenant ADD COLUMN stripe_customer_id VARCHAR;
```

---

## Métricas Trackeables por Tipo de Modelo

| Tipo | Training | Inference | Almacenamiento | Coste dominante |
|------|----------|-----------|----------------|-----------------|
| **Tabular (sklearn)** | CPU-time, RAM peak, n_features, n_samples, CV folds × metrics | Latencia p50/p95/p99, throughput (req/s), prediction drift | Model size (joblib), pipeline artifacts | CPU-hours |
| **Tabular (MLP)** | GPU-time, epochs, batch_size, loss curve, lr schedule | Latencia p50/p95/p99, batch throughput, uncertainty metrics | Model size (.pt/.ts), SHAP background | GPU-hours |
| **NLP (BERT et al.)** | GPU-time, tokens/sec, perplexity, seq_length, epochs | Latencia, **tokens in/out**, tokens/sec infer, TTFT | Model size (HF ~400MB), tokenizer | GPU-hours + tokens |
| **LLM (API-based)** | N/A (o fine-tuning tokens) | Tokens in/out, cost/request, latency TTFT + TBT | System prompt, few-shot examples | **$ tokens** (dominante) |
| **CV Classification** | GPU-time, images/sec, resolution, class balance | Latencia, images/sec throughput, confidence distribution | Model size, preprocessing cache | GPU-hours |
| **CV Segmentation** | GPU-time, IoU/Dice per epoch, image resolution | Latencia, output mask size, uncertainty heatmap | Model size, mask storage (.npy) | GPU-hours (alto) |
| **Time Series** | CPU-time, seasonal period, horizon, freq | Latencia, forecast horizon, confidence interval width | Model size, forecast cache | CPU-hours |

Atención especial a **LLM**: Es el único tipo donde el coste por inferencia domina sobre el coste de cómputo. Requiere tracking granular de tokens, caché semántica y presupuestos por tenant.

---

## Componentes del Sistema

### 0.1 UsageCollector — Middleware de métricas

Interceptor asíncrono que captura métricas de cada operación relevante.

```python
# app/observability/usage_collector.py
class UsageCollector:
    def record_inference(self, tenant_id, model_id, domain,
                         latency_ms, tokens_input=0, tokens_output=0,
                         success=True, tags=None):
        # → INSERT en usage_metrics
        # → Actualiza contadores Prometheus
        # → Si es LLM, calcula coste estimado vía CostCalculator

    def record_training(self, tenant_id, model_id, domain,
                        duration_s, gpu_hours=0, cpu_hours=0,
                        metrics=None, tags=None):
        # → INSERT en usage_metrics
        # → Actualiza cuota diaria de entrenamiento

    def record_storage(self, tenant_id, model_id, bytes, storage_type):
        # → Actualiza métricas Prometheus de almacenamiento
```

**Integración:** Se inyecta como dependencia FastAPI en los endpoints de inferencia y entrenamiento, y como middleware en InferenceService.

### 0.2 CostCalculator — Motor de estimación de costes

```python
# app/observability/cost_calculator.py

COST_TABLES = {
    "gpu_hour": {
        "A100": 3.50, "V100": 2.00, "T4": 0.80, "L4": 1.20,
    },
    "cpu_hour": 0.10,
    "storage_gb_month": 0.10,
    "tokens": {
        "gpt-4":        {"input": 0.03/1000, "output": 0.06/1000},
        "gpt-4-turbo":  {"input": 0.01/1000, "output": 0.03/1000},
        "gpt-3.5":      {"input": 0.0015/1000, "output": 0.002/1000},
        "claude-3-opus": {"input": 0.015/1000, "output": 0.075/1000},
        "claude-3-sonnet": {"input": 0.003/1000, "output": 0.015/1000},
    },
}


class CostCalculator:
    def estimate_inference_cost(self, model_id, domain,
                                latency_ms, tokens=None, gpu_type=None):
        """Estima coste de una inferencia basado en recursos consumidos."""

    def estimate_training_cost(self, gpu_hours, gpu_type, cpu_hours):
        """Estima coste de un entrenamiento."""

    def record_cost(self, tenant_id, cost_type, amount,
                    source, period_start, period_end, metadata=None):
        """Persiste un registro de coste en cost_records."""

    def get_monthly_cost(self, tenant_id):
        """Coste agregado del mes actual para un tenant."""

    def budget_remaining(self, tenant_id):
        """Presupuesto restante del tenant."""
```

### 0.3 Prometheus Custom Metrics

| Métrica | Tipo | Labels | Descripción |
|---------|------|--------|-------------|
| `praxisml_inference_latency_ms` | Histogram | tenant, domain, algorithm, success | Latencia de inferencia |
| `praxisml_inference_tokens_total` | Counter | tenant, model, direction | Conteo de tokens (NLP/LLM) |
| `praxisml_inference_cost_usd` | Counter | tenant, model, cost_type | Coste acumulado USD |
| `praxisml_training_duration_seconds` | Histogram | tenant, domain, algorithm | Duración de entrenamiento |
| `praxisml_training_cost_usd` | Counter | tenant, domain, algorithm | Coste de entrenamiento |
| `praxisml_model_storage_bytes` | Gauge | tenant, model | Tamaño del modelo en almacenamiento |
| `praxisml_ws_connections_active` | Gauge | tenant | Conexiones WebSocket activas |
| `praxisml_drift_score` | Gauge | model, metric | Último score de drift por modelo |
| `praxisml_quota_usage_pct` | Gauge | tenant, resource | % de uso de cuota |
| `praxisml_cost_budget_remaining` | Gauge | tenant | Presupuesto restante USD |
| `praxisml_predictions_total` | Counter | tenant, model, status | Conteo de predicciones |
| `praxisml_errors_total` | Counter | tenant, error_type, component | Errores por componente |

### 0.4 Grafana Dashboards (6 nuevos)

| Dashboard | Descripción | Paneles principales |
|-----------|-------------|-------------------|
| **Cost Center** | Costes agregados por tenant | Coste/día (bar chart), breakdown por modelo (pie), coste vs presupuesto (gauge), forecast mensual (line) |
| **Model Performance** | Métricas por modelo individual | Latencia p50/p95/p99 timeline, throughput, error rate, drift score histórico, uncertainty timeline |
| **Tenant Usage** | Uso global del tenant | Requests/día, entrenamientos/día, almacenamiento total, cuotas %, usuarios activos |
| **LLM Spend** | Costes específicos de LLM | Tokens input/output/día, coste por modelo, budget burn rate, coste por usuario |
| **Training Dashboard** | Métricas de entrenamiento | Duración por algoritmo, coste por entrenamiento, GPU utilization, success rate |
| **Drift Timeline** | Evolución temporal del drift | PSI/KS timeline por modelo, predicción de drift (forecast line), alertas |


### 0.5 Alertas y Notificaciones

**Canales soportados:** Webhook (Slack, Teams), Email, Dashboard badge, WebSocket push

**Alertas predefinidas:**

| Alerta | Trigger | Severidad |
|--------|---------|-----------|
| **Cost spike** | Coste/día > 2× promedio semanal (rolling 7d) | Alta |
| **Budget near limit** | Coste mensual > 80% del budget | Media |
| **Budget exceeded** | Coste mensual > 100% del budget | Crítica |
| **Latency degradation** | p99 latencia > 2× baseline semanal | Alta |
| **Latency spike** | p99 latencia > 5× baseline | Crítica |
| **Drift warning** | drift_score > threshold por primera vez | Media |
| **Drift critical** | drift_score > threshold por 3 chequeos consecutivos | Alta |
| **Token spike (LLM)** | Tokens/día > 3× promedio semanal | Alta |
| **Throughput drop** | throughput < 50% del promedio horario | Media |
| **Error rate spike** | error_rate > 5% en ventana 5 min | Crítica |
| **Entrenamiento fallido** | training_aborted_rate > 0 en ventana 1h | Media |
| **Quota imminent** | Uso de cuota > 90% | Baja |
| **Model zombie** | Modelo en Production sin inferencias en 7d | Baja |
| **GPU infrautilizada** | gpu_util < 20% por > 1h durante training | Media |

---

## API Endpoints

```
GET    /api/v1/observability/usage                         # Métricas agregadas del tenant
       ?tenant_id=X&from=...&to=...&domain=tabular&metric=latency_p99

GET    /api/v1/observability/usage/models/{model_id}       # Métricas de un modelo específico

GET    /api/v1/observability/costs                          # Desglose de costes
       ?tenant_id=X&period=monthly&cost_type=compute

GET    /api/v1/observability/costs/forecast                 # Predicción de coste mensual
       ?tenant_id=X

GET    /api/v1/observability/drift/history/{model_id}       # Serie temporal de drift

POST   /api/v1/observability/alerts                         # Crear alerta
GET    /api/v1/observability/alerts                         # Listar alertas del tenant
PATCH  /api/v1/observability/alerts/{id}                    # Actualizar alerta
DELETE /api/v1/observability/alerts/{id}                    # Eliminar alerta

GET    /api/v1/observability/models/{model_id}/health       # Model Health Score compuesto

GET    /api/v1/observability/tenant/{tenant_id}/report      # Reporte mensual PDF/CSV
```

---

## Plan de Implementación

| Sub-fase | Qué incluye | Dependencias | Esfuerzo | Prioridad |
|----------|------------|--------------|----------|-----------|
| **0.1** | Tablas SQL (`usage_metrics`, `cost_records`, `drift_history`, `alert_config`), clase `UsageCollector`, migraciones Alembic | Ninguna | 🟡 Medio (1-2 semanas) | **P0** |
| **0.2** | 12 custom Prometheus metrics, integración en training + inference + streaming | 0.1, training_service, inference_service | 🟡 Medio (2 semanas) | **P0** |
| **0.3** | `CostCalculator` con tablas de costes, registro automático post-inferencia y post-entrenamiento | 0.1, 0.2 | 🟡 Medio (1-2 semanas) | **P0** |
| **0.4** | 5 API endpoints REST (usage, costs, drift, alerts, health) | 0.1, 0.2, 0.3 | 🟡 Medio (2 semanas) | **P1** |
| **0.5** | 6 Grafana dashboards nuevos, paneles en dashboards existentes | 0.2, 0.3 | 🔴 Grande (2-3 semanas) | **P1** |
| **0.6** | Sistema de alertas con webhooks + emails, Celery Beat para chequeos periódicos | 0.2, 0.3, 0.5 | 🔴 Grande (3 semanas) | **P1** |
| **0.7** | Budget por tenant (`monthly_budget_usd`), billing cycle, forecast de coste, reportes PDF | 0.3, 0.6 | 🟢 Pequeño (1 semana) | **P2** |

**Secuencia recomendada:**

```
Q1 2027: 0.1 ──► 0.2 ──► 0.3
            │        │        │
            ▼        ▼        ▼
Q2 2027:  0.4 ──► 0.5 ──► 0.6
                            │
                            ▼
Q3 2027:                  0.7
```

---

## Coste y ROI Estimado

### Esfuerzo de implementación

| Recurso | Dedicación |
|---------|-----------|
| 1 backend senior | ~8 semanas total (distribuidas en 2Q) |
| 1 frontend/data | ~3 semanas (dashboards) |

### ROI Cualitativo

| Beneficio | Impacto |
|-----------|---------|
| Facturación por uso real a tenants | **Nuevo revenue stream** |
| Detección temprana de drift | Evita degradación de modelos en prod |
| Optimización de costes GPU | Ahorro 15-30% detectando infrautilización |
| Alertas proactivas | Reduce MTTR de 4h → 15min |
| Reportes a stakeholders | Visibilidad ejecutiva del valor de ML |

---

> Catálogo completo de métricas: [metrics-catalog.md](metrics-catalog.md)
> [← Volver al Roadmap principal](../roadmap.md)
