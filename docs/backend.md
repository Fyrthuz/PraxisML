# Backend Documentation (FastAPI)

El backend expone una API REST asíncrona robusta y maneja el motor de Machine Learning central. Está escrito en **Python 3.10+** usando `uv` para gestión de paquetes.

## 1. Stack Tecnológico
- **Framework Web**: `FastAPI` (asíncrono, validaciones automáticas, OpenAPI generada).
- **ORM**: `SQLAlchemy 2.0`
- **Gestión de Migraciones**: `Alembic`
- **Cola de Tareas**: `Celery` (Broker: Redis).
- **Control de Versiones y Tracking ML**: `MLflow` backend store.
- **Machine Learning**: `PyTorch`, `TorchScript`, `Scikit-learn`, `Pandas`, `Numpy`.
- **Integridad y Validaciones**: `Pydantic v2` / `Pydantic Settings`.
- **RBAC & Auth**: `PyJWT`, `passlib` (bcrypt), `slowapi`.

## 2. Estructura de Directorios Claves (`backend/app/`)

- **/api/routes/v1/**: Rutas REST (`tenants`, `models`, `datasets`, `training`, `predictions`, `preprocessing`, `profiling`, etc.).
- **/api/deps.py**: Inyección de dependencias (Sesiones DB, validaciones de tokens JWT, chequeo de roles RBAC, limitación de cuotas).
- **/core/**: Configuraciones maestras (`config.py`), jerarquía de excepciones (`exceptions.py`), logs estructurados (`logging.py`), motor de limitación (`rate_limit.py`) y seguridad JWT.
- **/core_ml/**: Motor central de predicciones e incertidumbre libre de dependencias HTTP. Define la capa base del inferenciador Scikit-learn y PyTorch, parseadores de datasets declarativos y familias de algoritmos.
- **/models/**: Declaraciones SQLAlchemy ORM.
- **/schemas/**: Validadores Pydantic de Request y Response.
- **services/**: Repositorios y adaptadores de servicios externos (Motor de almacenamiento MinIO, adaptador nativo de MLflow API, motor de validación `DataProfiler`).
- **/worker/**: Aplicación Celery asíncrona. Aquí se encuentran los `tasks` que toman el control computacional pesado.

## 3. Entidades Core

1. **Tenant**: Unidad lógica para suscripciones u organizaciones. Aloja las cuotas.
2. **User**: Entidad autenticada (AuthN). Pertenece a un Tenant.
3. **Dataset**: Binario ZIP junto con un `config.json` o subida declarativa.
4. **MLModel**: Referencia en BD local del modelo. Su binario real está en MLflow y su tracker UUID (`run_id`) está asignado de MLflow.
5. **Prediction**: Registro de ejecución e inferencia con resultados, tiempos y métricas de incertidumbre (ej. varianzas epistémicas).

## 4. Workflows Críticos

### 4.1. Carga de un Modelo (Safe TorchScript / Sklearn)
1. Usuario emite un request (POST) subiendo un `.pt` (PyTorch) o un job de MLflow existente (Entrenamiento Sklearn/PyTorch).
2. FastAPI intercepta, verifica RBAC (`editor`), comprueba Cuotas (`check_model_quota`).
3. El `MLflowService` interactúa con el demonio local de MLflow: registra el modelo creando un nuevo `Run`. Si es un upload, inyecta su binario allí.
4. FastAPI guarda en la base de datos local un registro asociando el UUID MLflow de este modelo con el Tenant.

### 4.2. Flujo de Entrenamiento (Async)
1. Usuario emite un request `/training/train`.
2. El request es rate-limitado (ej. 10 peticiones / min) por `slowapi`.
3. Validaciones de Cuota de Entrenamiento y Dataset asignado.
4. FastAPI crea un registro en tabla local como `RUNNING` o despacha un `task` Celery: `worker.delay(entrenamiento_args)`. Devuelve al cliente un `Task ID`.
5. El *Celery Worker* baja de Storage (MinIO) el dataset, aplica pipelines de Preprocesamiento (`ColumnTransformer` guardados), llama a Scikit-Learn de manera paralela (`n_jobs=-1`), y sube las métricas a MLflow. Al finalizar, actualiza DB.

### 4.3. Predicción e Incertidumbre
El motor `core_ml.factory` implementa Factory Pattern para inyectar la familia de inferenciadores y de estimación de incertidumbre.
1. Se despacha Tarea Async (Single Preprocesada/Batch).
2. Se selecciona método: ej. `mc_dropout`, `tta`, `noisy_inference`, `ensemble`.
3. Celery invoca la carga dinámica en memoria del binario (vía `torch.jit.load()` si es TorchScript impidiendo Pickle vulnerabilities, o cargando Pipelines/SKLearn si es ML general).
4. El Ensemble realiza N predicciones (Forward Passes simulados perturbando datos/modelo) y fusiona resultados antes de devolverlos.

## 5. Control de Acceso y Rate Limiting

- **Endpoints Exigentes (Train/Pred)**: Decorador `@limiter.limit` (ej. `30/minute`). Extrae la IP de la petición limitando abuso.
- **RBAC**: Las dependencias inyectadas resuelven si un usuario tiene la facultad. El primer usuario de un Tenant adquiere cargo de `ADMIN`. Existen los roles `admin`, `editor`, `viewer`.

```python
# Ejemplo inyección RBAC Pydantic
@router.delete("/{id}")
def delete_model(id: UUID, current_user = Depends(require_admin)):
    ...
```

- **Quotas**:
```python
@router.post("/")
def upload_dataset(..., verify_quota = Depends(check_dataset_quota)):
    ...
```

## 6. Pruebas y CI
El proyecto cuenta con suites de `pytest` (unitarias e integración).
- Uso intensivo de *mocking* (MagicMock) en Unit tests frente a bases de MLflow o MinIO.
- Base de datos en memoria o aislada transaccionalmente para Integration tests (API REST completa cubierta en `test_api_endpoints.py`).
- Flujo CI mediante GitHub Actions validando Lints (Ruff) con un límite de cobertura exigido en CI (Pytest `--cov-fail-under=30%` alcanzando ~45%).
