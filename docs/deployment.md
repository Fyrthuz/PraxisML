# Despliegue y Configuración (Deployment)

Guía completa para poner en marcha el stack completo en desarrollo, pruebas (Testing) o producción local mediante Docker Compose.

## 1. Stack Dockerizado
`docker-compose.yml` gestiona los contenedores base intercomunicados lógicamente por un switch local:

| Contenedor | Descripción / Base Image | Rol Maestro | Puerto (Host) |
|------------|-------------------------|-------------|---------------|
| `praxisml_db` | `pgvector/pgvector:pg15` | SOT: Metadatos de la API, de MLflow (schemas separados) y **embeddings semánticos (extensión `vector`)**. | `:5432` |
| `praxisml_redis` | `redis:7` | Celery broker y result backend. | `:6379` |
| `mlflow_server` | `mlflow:v3.10.0` | Servidor central de Tracking de Modelos de ML. | `:5000` |
| `praxisml_minio` | `minio:latest` | Objeto de almacenamiento compatible S3. | `:9000` / `:9001` (UI) |
| `praxisml_api` | `Dockerfile` custom (UV + Python) | Bouncer y Gateway de FastAPI. Inyecta seguridad de JWT. | `:8000` |
| `praxisml_worker` | `Dockerfile` custom (UV + Python) | Proceso en background para predicciones, inferencia y DL en Celery. | `--` |
| `praxisml_frontend`| `Dockerfile` Next.js React | Server Rendering e inicio de sesión SPA para cliente. | `:3000` |
| `praxisml_prometheus`| `prom/prometheus:latest` | Agente *Scraping* de métricas para FastAPI y Worker. | `:9090` |
| `praxisml_grafana` | `grafana/grafana:latest` | Componente de Visualización en vivo (Dashboard por API / OS). | `:3001` |
| `praxisml_vllm` *(opcional)* | `vllm/vllm-openai:latest` | Servidor LLM OpenAI-compatible para el asistente RAG (perfil `nlp`, requiere GPU). | `:8001` |

*(El contenedor `praxisml_minio_init` es efímero y simplemente crea el Bucket predeterminado si no existe)*.

> **NLP (Fase 1):** la base de datos usa la imagen `pgvector/pgvector:pg15` (Postgres 15 + extensión `vector`). La extensión se crea en `infra/postgres/init-mlflow-db.sh` y en la migración `phase1_nlp_001`. El servicio `vllm` se activa con `docker compose --profile nlp up -d vllm`.

### Migración `postgres:15` → `pgvector/pgvector:pg15`

Al cambiar la imagen con un volumen existente puede aparecer en los logs del contenedor `praxisml_db`:

```
WARNING: database "praxisml_db" has a collation version mismatch
DETAIL: The database was created using collation version 2.41, but the operating system provides version 2.36.
```

Es **benigno** (los datos no se corrompen), pero conviene refrescar las versiones de collation para silenciarlo y evitar riesgo en `ORDER BY` sobre texto:

```bash
for db in postgres praxisml_db mlflow_db template1; do
  docker exec praxisml_db psql -U postgres -d "$db" -c "ALTER DATABASE $db REFRESH COLLATION VERSION;"
done
```

Si prefieres empezar de cero (solo dev): `docker compose down -v && docker compose up --build`.

## 2. Variables de Entorno (`.env`)

El backend confía plenamente en `pydantic-settings`.
Existen 3 niveles de precedencia (Del menor al mayor):
1. Defaults internos de código (Solo locales).
2. Fichero oculto local `.env` ubicado en la raíz, leído en memoria por Python o Docker.
3. Exportación explícita (Bash exports o Inyectores Docker/Kuber).

### Ejemplos Críticos (`.env.example` clonado):
```env
# Define si corres con Logging DEBUG (development) o INFO estricto (production)
ENVIRONMENT=production

# Semilla asimétrica que corrobora la firma criptográfica del Auth. 
# En prod la app colapsa un 500 error si es la por defecto.
SECRET_KEY=clave_ultra_secreta_aqui...

# Selecciona hacia dónde escriben los modelos exportables o ZIPs: 'local', 'minio' o remotas en 's3' real de Amazon AWS.
STORAGE_BACKEND=minio

# ── NLP / LLM (Fase 1) ──────────────────────────────────────────────────
# Proveedor LLM: vllm | openai | anthropic | ollama
LLM_PROVIDER=vllm
VLLM_BASE_URL=http://vllm:8001/v1
VLLM_MODEL_NAME=Qwen/Qwen2.5-7B-Instruct
LLM_API_KEY=EMPTY
EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
```

### Configuración de vLLM

- El modelo servido (`VLLM_MODEL_NAME`) **debe coincidir** con el `resource_id` de precios en `resource_pricing` (categoría `llm_tokens`) para que el coste se compute correctamente. El seed inicial registra el modelo por defecto a precio 0.0 (self-hosted); se ajusta vía `POST /api/v1/observability/pricing` (admin).
- Para desplegar el LLM: `docker compose --profile nlp up -d vllm` (requiere GPU; el `VLLM_MODEL` se descarga al primer arranque).

## 3. Comandos Esenciales de Operación

1. **Clean Start**: Detiene todo destruyendo volúmenes (Limpieza profunda para DB/Redis/S3).
   ```bash
   docker-compose down -v && docker-compose up --build
   ```

2. **Backend Local Puro** (Sin worker en Docker, usado para debugear código FastAPI rápido):
   ```bash
   # Asegurar los dependientes base.
   docker-compose up -d db redis mlflow minio minio-init
   
   cd backend
   uv sync
   uv run python migrate.py
   uv run uvicorn app.main:app --reload
   ```

3. **Ejecutar Pruebas Automatizadas (Celery + Integración API REST)**
   El CLI de `uv` es el manejador de este entorno, no `pip` estándar:
   ```bash
   cd backend
   uv run pytest tests/ -v --cov=app --cov-fail-under=30
   uv run ruff check app/ --select=E,F,W --ignore=E501
   ```

## 4. Persistencia (Volúmenes de Docker)

El sistema genera una serie de *named volumes* mapeados globalmente:
- `postgres_data`: Mantenimiento seguro para base de datos (incluye los índices `pgvector`).
- `minio_data`: Aislado seguro para objetos `.pt`, `models` en bruto y `datasets`.
- `grafana_data`: Provisión del Dashboard configurado o configuraciones de accesos.
- `prometheus_data`: Scraping o Time-series index files (TSDB) sin truncamientos tras reinicios.
- `vllm_cache`: Cache de modelos HuggingFace descargados por el servidor vLLM (Fase 1).

**Directorio Local (`/backend/data`)**: Montado solo para desarrollos nativos y para inicializar o explorar los directorios en host sin recurrir a accesos por MinIO Console.

## 5. Visualización / Telemetría (Grafana)

Por defecto, existe un panel de Telemetría (pre-importado vía carpeta `/provisioning/dashboards/*.json`) dentro de **http://localhost:3001** (Acceso nativo `admin:admin`).
Analiza `API (Throughput/Durations)`, conteo global de incidencias/Rendimiento en ML, Picos de 5xx o Bloqueos y Fallas AuthN 403.
Punto base del Scraping: **http://localhost:8000/metrics**.
