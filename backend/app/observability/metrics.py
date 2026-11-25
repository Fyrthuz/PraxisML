"""
12 métricas custom de Prometheus para observabilidad.

Todas siguen el prefijo `praxisml_` para facilitar
la identificación en Grafana y alertas.
"""

from prometheus_client import Counter, Gauge, Histogram

# ── Inferencia ──────────────────────────────────────────────────────────────────

inference_latency_ms = Histogram(
    "praxisml_inference_latency_ms",
    "Latencia de inferencia en milisegundos",
    ["tenant_id", "domain", "algorithm", "success"],
    buckets=[10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000],
)

inference_tokens_total = Counter(
    "praxisml_inference_tokens_total",
    "Tokens procesados en inferencia (input/output)",
    ["tenant_id", "model_id", "direction"],
)

inference_cost_usd = Counter(
    "praxisml_inference_cost_usd",
    "Coste acumulado de inferencia en USD",
    ["tenant_id", "model_id", "cost_type"],
)

predictions_total = Counter(
    "praxisml_predictions_total",
    "Total de predicciones ejecutadas",
    ["tenant_id", "model_id", "status"],
)

# ── Entrenamiento ───────────────────────────────────────────────────────────────

training_duration_seconds = Histogram(
    "praxisml_training_duration_seconds",
    "Duración de entrenamiento en segundos",
    ["tenant_id", "domain", "algorithm"],
    buckets=[10, 30, 60, 120, 300, 600, 1800, 3600, 7200],
)

training_cost_usd = Counter(
    "praxisml_training_cost_usd",
    "Coste acumulado de entrenamiento en USD",
    ["tenant_id", "domain", "algorithm"],
)

# ── Almacenamiento ──────────────────────────────────────────────────────────────

model_storage_bytes = Gauge(
    "praxisml_model_storage_bytes",
    "Tamaño del modelo en almacenamiento (bytes)",
    ["tenant_id", "model_id"],
)

# ── WebSocket ───────────────────────────────────────────────────────────────────

ws_connections_active = Gauge(
    "praxisml_ws_connections_active",
    "Conexiones WebSocket activas",
    ["tenant_id"],
)

# ── Drift ───────────────────────────────────────────────────────────────────────

drift_score = Gauge(
    "praxisml_drift_score",
    "Valor de drift por modelo y métrica (PSI/KS/JS)",
    ["model_id", "metric"],
)

# ── Cuotas ──────────────────────────────────────────────────────────────────────

quota_usage_pct = Gauge(
    "praxisml_quota_usage_pct",
    "Porcentaje de uso de cuota por tenant y recurso",
    ["tenant_id", "resource"],
)

# ── Presupuesto ─────────────────────────────────────────────────────────────────

cost_budget_remaining = Gauge(
    "praxisml_cost_budget_remaining",
    "Presupuesto mensual restante en USD",
    ["tenant_id"],
)

# ── Errores ─────────────────────────────────────────────────────────────────────

errors_total = Counter(
    "praxisml_errors_total",
    "Errores capturados por componente y tipo",
    ["tenant_id", "error_type", "component"],
)

# ── NLP Engine (Fase 1) ──────────────────────────────────────────────────────────

nlp_inference_latency_ms = Histogram(
    "praxisml_nlp_inference_latency_ms",
    "Latencia de inferencia NLP en milisegundos",
    ["tenant_id", "model_id", "task_type"],
    buckets=[10, 25, 50, 100, 250, 500, 1000, 2500, 5000],
)

nlp_predictions_total = Counter(
    "praxisml_nlp_predictions_total",
    "Total de predicciones NLP por tarea y estado",
    ["tenant_id", "model_id", "task_type", "status"],
)

embeddings_total = Counter(
    "praxisml_embeddings_total",
    "Operaciones de embeddings por tenant (embed|index|search)",
    ["tenant_id", "operation"],
)

semantic_search_latency_ms = Histogram(
    "praxisml_semantic_search_latency_ms",
    "Latencia de búsqueda semántica en milisegundos",
    ["tenant_id", "index_type"],
    buckets=[10, 25, 50, 100, 250, 500, 1000, 2500, 5000],
)

# ── LLM Gateway (vLLM / OpenAI / Ollama) ─────────────────────────────────────────

llm_tokens_total = Counter(
    "praxisml_llm_tokens_total",
    "Tokens procesados por el LLM (input/output)",
    ["tenant_id", "model_id", "provider", "direction"],
)

llm_requests_total = Counter(
    "praxisml_llm_requests_total",
    "Requests al LLM por proveedor y estado",
    ["tenant_id", "model_id", "provider", "status"],
)

llm_latency_ms = Histogram(
    "praxisml_llm_latency_ms",
    "Latencia de las llamadas al LLM en milisegundos",
    ["tenant_id", "model_id", "provider"],
    buckets=[100, 250, 500, 1000, 2500, 5000, 10000, 30000, 60000],
)

# ── Model Serving Endpoints & User Pipelines ────────────────────────────────────

serving_requests_total = Counter(
    "praxisml_serving_requests_total",
    "Total de invocaciones a Model Serving Endpoints por slug, variante y estado",
    ["tenant_id", "endpoint_slug", "variant", "status"],
)

serving_latency_ms = Histogram(
    "praxisml_serving_latency_ms",
    "Latencia de invocaciones a Model Serving Endpoints en milisegundos",
    ["tenant_id", "endpoint_slug", "runtime"],
    buckets=[5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000],
)

pipeline_runs_total = Counter(
    "praxisml_pipeline_runs_total",
    "Total de ejecuciones de UserPipelines por engine y estado",
    ["tenant_id", "engine", "status"],
)

pipeline_duration_seconds = Histogram(
    "praxisml_pipeline_duration_seconds",
    "Duración de ejecuciones de UserPipelines en segundos",
    ["tenant_id", "engine"],
    buckets=[0.1, 0.5, 1, 2, 5, 15, 30, 60, 180],
)

# ── Platform Services & Queues Health ─────────────────────────────────────────

service_health_status = Gauge(
    "praxisml_service_health",
    "Estado de salud de los servicios de la plataforma (1=ok, 0.5=idle/rag_local, 0=offline/error)",
    ["service"],
)

celery_queue_depth = Gauge(
    "praxisml_celery_queue_depth",
    "Profundidad actual de cada cola de tareas Celery en Redis",
    ["queue"],
)

