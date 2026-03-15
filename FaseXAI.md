# Fase XAI: Plan de Implementación

## Resumen
Este documento detalla el plan de implementación para optimizar el sistema de modelos de datos tabulares, incluyendo:
1. Corrección de Pipeline y Registro de Datos (DVC)
2. Explicabilidad Tabular (XAI) con SHAP
3. Data Drift y Alertas
4. Endpoint de Predicción en Streaming con WebSockets

---

## 1. Corrección de Pipeline y Registro de Datos (DVC)

### Objetivo
Asegurar que cada vez que se ejecute el pipeline de preprocesamiento, el dataset resultante se registre automáticamente en DVC, y persistir el identificador de DVC en los metadatos del modelo.

### Tareas

#### 1.1 Modificar preprocessing.py para registrar dataset en DVC
**Archivo:** `backend/app/api/routes/v1/preprocessing.py`
**Acción:** Después de guardar el dataset transformado, llamar a `track_dataset_with_dvc()` y actualizar campos DVC en el nuevo dataset.

```python
# En la función apply_preprocessing, después de df_transformed.to_csv():
from app.services.dvc_service import track_dataset_with_dvc

# Registrar en DVC
dvc_info = track_dataset_with_dvc(
    tenant_id=tenant.id,
    file_path=new_file_path,
    registry_name=f"preprocessed_{dataset.name}"
)

# Actualizar campos DVC en el nuevo dataset
new_dataset.is_dvc_tracked = dvc_info.get("is_dvc_tracked", False)
new_dataset.dvc_hash = dvc_info.get("dvc_hash")
new_dataset.dvc_remote = dvc_info.get("dvc_remote")
new_dataset.dvc_registry_name = dvc_info.get("dvc_registry_name")
new_dataset.dvc_version = dvc_info.get("dvc_version")
```

#### 1.2 Añadir campo dataset_dvc_hash en MLModel (migración)
**Archivo:** `backend/app/models/ml_model.py`
**Acción:** Añadir campos para trazabilidad DVC.

```python
# Añadir en clase MLModel:
dataset_dvc_hash = Column(String, nullable=True)
dataset_dvc_registry_name = Column(String, nullable=True)
```

**Migración Alembic:** Crear nueva migración para añadir estos campos.

#### 1.3 Modificar train.py para guardar dvc_hash en el modelo
**Archivo:** `backend/app/worker/tasks/train.py`
**Acción:** Al registrar el modelo, guardar el DVC hash del dataset en los campos nuevos.

```python
# En la creación de new_model:
new_model = MLModel(
    # ... otros campos ...
    dataset_dvc_hash=dataset.dvc_hash,
    dataset_dvc_registry_name=dataset.dvc_registry_name,
)
```

---

## 2. Explicabilidad Tabular (XAI) para todos los algoritmos

### Objetivo
Soportar SHAP con KernelExplainer para todos los algoritmos (árboles, lineales, SVM, KNN, PyTorch MLP). Cálculo al vuelo. Modelos PyTorch cargados desde MLflow.

### Tareas

#### 2.1 Crear módulo explainability.py con SHAP
**Archivo:** `backend/app/core_ml/explainability.py` (nuevo)
**Acción:** Crear módulo con `shap.KernelExplainer` para todos los modelos.

```python
import shap
import numpy as np
from typing import Any, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

def get_shap_values(model: Any, data: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
    """
    Calcula SHAP values usando KernelExplainer para cualquier modelo.
    
    Args:
        model: Modelo scikit-learn o PyTorch (con método predict)
        data: Datos de entrada (numpy array)
        feature_names: Nombres de las características
    
    Returns:
        Dict con 'shap_values' (lista de arrays) y 'expected_value'
    """
    try:
        # Crear wrapper para modelo PyTorch si es necesario
        if hasattr(model, 'predict'):
            # Modelo scikit-learn
            predictor = model.predict
        else:
            # Modelo PyTorch desde MLflow (debe tener método predict)
            if not hasattr(model, 'predict'):
                raise ValueError("Modelo no tiene método predict")
            predictor = model.predict
        
        # Crear background dataset para KernelExplainer (muestra de datos)
        background = shap.sample(data, 100) if len(data) > 100 else data
        
        # Crear explainer
        explainer = shap.KernelExplainer(predictor, background)
        
        # Calcular SHAP values
        shap_values = explainer.shap_values(data)
        
        return {
            'shap_values': shap_values.tolist() if isinstance(shap_values, np.ndarray) else [sv.tolist() for sv in shap_values],
            'expected_value': explainer.expected_value,
            'feature_names': feature_names
        }
    except Exception as e:
        logger.error(f"Error calculando SHAP values: {e}")
        raise
```

#### 2.2 Añadir SHAP a dependencias
**Archivo:** `backend/pyproject.toml`
**Acción:** Añadir dependencia `shap`.

```toml
# En dependencies:
"shap>=0.44.0",
```

#### 2.3 Modificar endpoints de predicción para aceptar explain=true
**Archivo:** `backend/app/api/routes/v1/predictions.py`
**Acción:** Añadir parámetro `explain: bool = False` a todos los endpoints.

**Endpoint 1: Predicción individual**
```python
class PredictionRequest(BaseModel):
    dataset_id: str
    model_id: str
    uncertainty_method: str = "mc_dropout"
    explain: bool = False  # NUEVO
```

**Endpoint 2: Predicción single tabular**
```python
@router.post("/predictions/predict/single")
def request_single_prediction(
    # ... otros parámetros ...
    explain: bool = Form(False),  # NUEVO
):
```

**Endpoint 3: Predicción batch**
```python
@router.post("/predictions/predict/batch")
def request_batch_prediction(
    # ... otros parámetros ...
    explain: bool = Form(False),  # NUEVO
):
```

#### 2.4 Modificar tareas de predicción para calcular SHAP values
**Archivo:** `backend/app/worker/tasks/single_predict.py`
**Acción:** Calcular SHAP values cuando `explain=True`.

```python
# En la función run_single_tabular_inference:
# Después de obtener result_dict:
if explain:  # NUEVO
    from app.core_ml.explainability import get_shap_values
    try:
        # Obtener feature_names del modelo
        feature_names = ml_model.metrics_metadata.get("feature_names", [])
        if not feature_names:
            feature_names = [f"feature_{i}" for i in range(input_data.shape[1])]
        
        # Calcular SHAP values
        shap_result = get_shap_values(model, input_data, feature_names)
        result_dict['shap_values'] = shap_result['shap_values']
        result_dict['shap_expected_value'] = shap_result['expected_value']
        result_dict['feature_names'] = feature_names
    except Exception as e:
        logger.warning(f"No se pudieron calcular SHAP values: {e}")
```

**Archivo:** `backend/app/worker/tasks/predict.py` (similar para batch)

#### 2.5 Crear endpoint para devolver SHAP values
**Archivo:** `backend/app/api/routes/v1/predictions.py`
**Acción:** Crear endpoint `GET /predictions/{prediction_id}/explain`.

```python
@router.get("/predictions/{prediction_id}/explain")
def get_prediction_explain(
    prediction_id: str,
    _user: User = Depends(require_viewer),
    tenant: Tenant = Depend(get_current_tenant),
    db: Session = Depends(get_db),
):
    """
    Devuelve los SHAP values calculados para una predicción.
    Calcula al vuelo si no están almacenados.
    """
    prediction = db.query(Prediction).filter(
        Prediction.id == prediction_id, Prediction.tenant_id == tenant.id
    ).first()
    
    if not prediction:
        raise HTTPException(status_code=404, detail="Predicción no encontrada")
    
    # Si ya se calcularon y están en result_path, cargarlos
    # Si no, calcular al vuelo usando el modelo y datos de entrada
    # (Implementación específica según estructura de datos)
    
    return {"shap_values": [...], "feature_names": [...]}
```

#### 2.6 Crear componente ExplainabilityPanel en frontend
**Archivo:** `frontend/frontend/src/components/ExplainabilityPanel.tsx` (nuevo)
**Acción:** Componente React que visualice contribuciones de variables con gráficos de barras.

```tsx
import React from 'react';
import { Bar } from 'react-chartjs-2';

interface ExplainabilityPanelProps {
  shapValues: number[];
  featureNames: string[];
  prediction: number;
}

export const ExplainabilityPanel: React.FC<ExplainabilityPanelProps> = ({
  shapValues,
  featureNames,
  prediction
}) => {
  const data = {
    labels: featureNames,
    datasets: [{
      label: 'Impacto en predicción',
      data: shapValues,
      backgroundColor: shapValues.map(v => v >= 0 ? 'rgba(75, 192, 192, 0.6)' : 'rgba(255, 99, 132, 0.6)'),
    }]
  };
  
  return (
    <div className="explainability-panel">
      <h3>Explicabilidad (SHAP)</h3>
      <Bar data={data} />
      <p>Predicción: {prediction.toFixed(4)}</p>
    </div>
  );
};
```

---

## 3. Data Drift y Alertas

### Objetivo
Monitorizar drift entre dataset de entrenamiento y datos de producción con umbrales configurables por dataset/modelo (herencia modelo → dataset).

### Tareas

#### 3.1 Añadir evidently a dependencias
**Archivo:** `backend/pyproject.toml`
**Acción:** Añadir dependencia `evidently`.

```toml
# En dependencies:
"evidently>=0.4.0",
```

#### 3.2 Crear servicio de drift
**Archivo:** `backend/app/services/drift_service.py` (nuevo)
**Acción:** Crear servicio para comparar datasets usando Evidently.

```python
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.metrics import ColumnDriftMetric
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class DriftService:
    def __init__(self, tenant_id: str):
        self.tenant_id = tenant_id
    
    def calculate_drift(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        psi_threshold: float = 0.2,
        ks_threshold: float = 0.05
    ) -> Dict[str, Any]:
        """
        Calcula drift entre dataset de referencia (entrenamiento) y datos actuales (producción).
        """
        try:
            column_mapping = ColumnMapping()
            
            report = Report(metrics=[
                DataDriftPreset(),
            ])
            
            report.run(reference_data=reference_data, current_data=current_data, column_mapping=column_mapping)
            
            result = report.as_dict()
            
            # Extraer métricas de drift
            drift_metrics = {
                'dataset_drift': result['metrics'][0]['result']['dataset_drift'],
                'drift_by_columns': result['metrics'][0]['result']['drift_by_columns'],
                'psi_threshold': psi_threshold,
                'ks_threshold': ks_threshold,
            }
            
            return drift_metrics
            
        except Exception as e:
            logger.error(f"Error calculando drift: {e}")
            raise
```

#### 3.3 Crear tarea Celery programada para drift
**Archivo:** `backend/app/worker/tasks/drift_check.py` (nuevo)
**Acción:** Tarea Celery para ejecutar análisis de drift diariamente.

```python
from celery import Task
from app.worker.celery_app import celery_app
from app.services.drift_service import DriftService
from app.models.dataset import Dataset
from app.models.ml_model import MLModel
from app.database import SessionLocal
import logging

logger = logging.getLogger(__name__)

@celery_app.task(bind=True, name="app.worker.tasks.drift_check.run_drift_check")
def run_drift_check(self: Task, model_id: str):
    """
    Ejecuta análisis de drift para un modelo específico.
    """
    db = SessionLocal()
    try:
        model = db.query(MLModel).filter(MLModel.id == model_id).first()
        if not model:
            raise ValueError(f"Modelo {model_id} no encontrado")
        
        # Obtener dataset de entrenamiento (referencia)
        # Implementar lógica para cargar dataset de entrenamiento
        
        # Obtener datos de producción (ej: últimos N días)
        # Implementar lógica para cargar datos de producción
        
        # Calcular umbrales (heredar de dataset si no específicos)
        psi_threshold = model.psi_threshold or 0.2
        ks_threshold = model.ks_threshold or 0.05
        
        # Calcular drift
        drift_service = DriftService(model.tenant_id)
        drift_result = drift_service.calculate_drift(
            reference_data=reference_data,
            current_data=current_data,
            psi_threshold=psi_threshold,
            ks_threshold=ks_threshold
        )
        
        # Guardar resultado en BD o notificar si drift detectado
        if drift_result['dataset_drift']:
            logger.warning(f"Drift detectado en modelo {model_id}")
            # Implementar lógica de alertas
        
        return drift_result
        
    except Exception as e:
        logger.error(f"Error en drift check: {e}")
        raise
    finally:
        db.close()
```

#### 3.4 Crear endpoint GET para reporte de drift
**Archivo:** `backend/app/api/routes/v1/drift.py` (nuevo)
**Acción:** Endpoint `GET /drift/report/{model_id}` o `/{dataset_id}`.

```python
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database import get_db
from app.models.ml_model import MLModel
from app.models.dataset import Dataset
from app.api.deps import get_current_tenant, require_viewer
from app.services.drift_service import DriftService

router = APIRouter()

@router.get("/drift/report/{model_id}")
def get_drift_report(
    model_id: str,
    _user: User = Depends(require_viewer),
    tenant: Tenant = Depends(get_current_tenant),
    db: Session = Depends(get_db),
):
    """
    Devuelve reporte de drift para un modelo específico.
    """
    model = db.query(MLModel).filter(
        MLModel.id == model_id, MLModel.tenant_id == tenant.id
    ).first()
    
    if not model:
        raise HTTPException(status_code=404, detail="Modelo no encontrado")
    
    # Implementar lógica para cargar reporte de drift
    # (puede ser de una ejecución reciente o calcular al vuelo)
    
    return {"model_id": model_id, "drift_report": {...}}
```

#### 3.5 Configurar scheduling (Celery beat)
**Archivo:** `backend/app/worker/celery_app.py`
**Acción:** Configurar tarea programada para drift.

```python
from celery.schedules import crontab

app.conf.beat_schedule = {
    'run-drift-check-daily': {
        'task': 'app.worker.tasks.drift_check.run_drift_check',
        'schedule': crontab(hour=2, minute=0),  # Ejecutar a las 2 AM diariamente
        'args': (),
    },
}
```

#### 3.6 Añadir campos de umbrales de drift
**Archivo:** `backend/app/models/dataset.py` y `ml_model.py`
**Acción:** Añadir campos para umbrales configurables.

```python
# En Dataset:
psi_threshold = Column(Float, nullable=True, default=0.2)
ks_threshold = Column(Float, nullable=True, default=0.05)

# En MLModel:
psi_threshold = Column(Float, nullable=True)
ks_threshold = Column(Float, nullable=True)
```

#### 3.7 Endpoint para actualizar umbrales de drift
**Archivo:** `backend/app/api/routes/v1/datasets.py` y `models.py`
**Acción:** Endpoint PATCH para actualizar umbrales.

```python
@router.patch("/datasets/{dataset_id}/drift-thresholds")
def update_dataset_drift_thresholds(
    dataset_id: str,
    psi_threshold: Optional[float] = None,
    ks_threshold: Optional[float] = None,
    # ...
):
    # Implementar lógica de actualización
```

---

## 4. Endpoint de Predicción en Streaming con WebSockets

### Objetivo
Permitir inferencia continua mediante WebSockets, enviando predicciones + incertidumbre + SHAP (si `explain=true`) en tiempo real. Autenticación JWT y cuota de 10 conexiones por tenant.

### Tareas

#### 4.1 Crear endpoint WebSocket para streaming de predicciones
**Archivo:** `backend/app/api/routes/v1/streaming.py` (nuevo)
**Acción:** Crear endpoint WebSocket con autenticación JWT.

```python
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, Query
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
import json
import logging
from typing import Dict

from app.database import get_db
from app.models.ml_model import MLModel
from app.models.user import User
from app.api.deps import get_current_tenant, require_editor
from app.core.security import decode_token
from app.core_ml.explainability import get_shap_values
from app.worker.tasks.streaming_predict import process_row

logger = logging.getLogger(__name__)
router = APIRouter()

# Almacenar conexiones activas por tenant
active_connections: Dict[str, Dict[str, WebSocket]] = {}

@router.websocket("/streaming/predict/{model_id}")
async def websocket_predict(
    websocket: WebSocket,
    model_id: str,
    token: str = Query(..., description="JWT token"),
    explain: bool = Query(False, description="Incluir explicabilidad SHAP"),
    db: Session = Depends(get_db),
):
    """
    Endpoint WebSocket para predicciones en tiempo real.
    Autenticación JWT via query parameter ?token=...
    """
    # Validar JWT
    payload = decode_token(token)
    if not payload:
        await websocket.close(code=1008, reason="Invalid token")
        return
    
    user_id = payload.get("sub")
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        await websocket.close(code=1008, reason="User not found")
        return
    
    tenant_id = user.tenant_id
    
    # Verificar cuota de conexiones WebSocket
    tenant = db.query(Tenant).filter(Tenant.id == tenant_id).first()
    if tenant.max_websocket_connections is not None:
        current_connections = len(active_connections.get(tenant_id, {}))
        if current_connections >= tenant.max_websocket_connections:
            await websocket.close(code=1008, reason="WebSocket quota exceeded")
            return
    
    # Conectar
    await websocket.accept()
    
    # Almacenar conexión
    if tenant_id not in active_connections:
        active_connections[tenant_id] = {}
    connection_id = f"{model_id}_{len(active_connections[tenant_id])}"
    active_connections[tenant_id][connection_id] = websocket
    
    try:
        # Obtener modelo
        model = db.query(MLModel).filter(
            MLModel.id == model_id, MLModel.tenant_id == tenant_id
        ).first()
        
        if not model:
            await websocket.send_json({"error": "Model not found"})
            await websocket.close()
            return
        
        # Loop principal: recibir filas y enviar predicciones
        while True:
            data = await websocket.receive_text()
            row_data = json.loads(data)
            
            # Procesar fila individual
            result = await process_row(model, row_data, explain)
            
            # Enviar respuesta
            await websocket.send_json(result)
            
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {connection_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close(code=1011, reason=str(e))
    finally:
        # Limpiar conexión
        if tenant_id in active_connections and connection_id in active_connections[tenant_id]:
            del active_connections[tenant_id][connection_id]
```

#### 4.2 Modificar tareas de predicción para soportar streaming
**Archivo:** `backend/app/worker/tasks/streaming_predict.py` (nuevo)
**Acción:** Función para procesar filas individuales.

```python
import torch
import numpy as np
import pandas as pd
from typing import Dict, Any
import logging

from app.core_ml.factory import PredictionFactory, UncertaintyMethod
from app.core_ml.preprocessing import load_pipeline, apply_pipeline
from app.core_ml.explainability import get_shap_values

logger = logging.getLogger(__name__)

async def process_row(model: Any, row_data: Dict[str, Any], explain: bool = False) -> Dict[str, Any]:
    """
    Procesa una fila individual y devuelve predicción + incertidumbre + SHAP (si corresponde).
    """
    try:
        # 1. Cargar modelo si es necesario (caché local)
        # 2. Aplicar preprocesamiento si existe
        # 3. Calcular predicción
        # 4. Calcular incertidumbre
        # 5. Calcular SHAP si explain=True
        
        # Ejemplo básico:
        feature_names = model.metrics_metadata.get("feature_names", [])
        if not feature_names:
            feature_names = list(row_data.keys())
        
        # Convertir a DataFrame
        df = pd.DataFrame([row_data])
        
        # Aplicar preprocesamiento
        if model.preprocessing_pipeline_path:
            pipeline = load_pipeline(model.preprocessing_pipeline_path)
            df, _ = apply_pipeline(pipeline, df)
        
        input_data = df.to_numpy().astype(np.float32)
        
        # Cargar modelo (simplificado - en producción usar caché)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # ... cargar modelo desde MLflow ...
        
        # Calcular predicción
        # ... lógica de predicción ...
        
        result = {
            "prediction": prediction,
            "uncertainty": uncertainty,
        }
        
        if explain:
            shap_result = get_shap_values(model, input_data, feature_names)
            result["shap_values"] = shap_result["shap_values"]
            result["feature_names"] = feature_names
        
        return result
        
    except Exception as e:
        logger.error(f"Error procesando fila: {e}")
        return {"error": str(e)}
```

#### 4.3 Añadir campo max_websocket_connections en Tenant
**Archivo:** `backend/app/models/tenant.py`
**Acción:** Añadir campo para cuota de conexiones WebSocket.

```python
class Tenant(Base):
    # ... campos existentes ...
    max_websocket_connections = Column(Integer, nullable=True, default=10)
```

**Migración Alembic:** Crear migración para añadir este campo.

#### 4.4 Crear función check_websocket_quota
**Archivo:** `backend/app/api/deps.py`
**Acción:** Verificar cuota de conexiones WebSocket.

```python
def check_websocket_quota(
    tenant: Tenant = Depends(get_current_tenant),
) -> Tenant:
    """Verifica que el tenant no haya excedido su cuota de conexiones WebSocket."""
    # La verificación se hace en el endpoint WebSocket
    return tenant
```

#### 4.5 Manejo de conexiones concurrentes
**Archivo:** `backend/app/api/routes/v1/streaming.py`
**Acción:** Implementar diccionario de conexiones y limpieza automática.

(Ver implementación en 4.1)

#### 4.6 Documentación API
**Archivo:** Docstring en endpoint WebSocket y actualización de OpenAPI.

---

## 5. Autenticación JWT para WebSockets

### Tarea
Validar JWT al conectar WebSocket.

#### 5.1 Extraer y validar token
**Archivo:** `backend/app/api/routes/v1/streaming.py`
**Acción:** Extraer token de query string o header.

```python
# En websocket_predict:
# Token de query string: ?token=...
payload = decode_token(token)
if not payload:
    await websocket.close(code=1008, reason="Invalid token")
    return
```

#### 5.2 Asegurar decode_token funciona
**Archivo:** `backend/app/core/security.py`
**Acción:** Verificar función `decode_token`.

```python
def decode_token(token: str) -> Optional[dict]:
    """Decodifica token JWT."""
    try:
        # Implementación existente
        # ...
        return payload
    except Exception:
        return None
```

---

## Cronograma de Implementación

1. **Semana 1:** Tareas 1.1-1.3 (DVC)
2. **Semana 2:** Tareas 2.1-2.3 (XAI base)
3. **Semana 3:** Tareas 2.4-2.6 (XAI completo + frontend)
4. **Semana 4:** Tareas 3.1-3.5 (Drift)
5. **Semana 5:** Tareas 4.1-4.6 (WebSockets)
6. **Semana 6:** Tareas 5.1-5.2 (Autenticación) + Integración y pruebas

---

## Notas Importantes

- Todas las migraciones de base de datos deben crearse con Alembic.
- Los endpoints WebSocket deben incluir manejo de errores robusto.
- SHAP puede ser computacionalmente intensivo; considerar límites de tiempo.
- Para producción, implementar caché de modelos para WebSockets.
- Documentar cada cambio en el README del proyecto.

Fin del plan de implementación.