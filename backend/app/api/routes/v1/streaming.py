"""
Endpoint WebSocket para predicciones en tiempo real.
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
from sqlalchemy.orm import Session
import json
import logging
from typing import Dict

from app.database import get_db
from app.models.ml_model import MLModel
from app.models.user import User
from app.models.tenant import Tenant
from app.api.deps import get_current_tenant, require_editor
from app.core.security import decode_token
from app.core_ml.explainability import get_shap_values
import torch
import numpy as np
import pandas as pd
from app.core_ml.factory import PredictionFactory, UncertaintyMethod
from app.core_ml.preprocessing import load_pipeline, apply_pipeline

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
    # No tenemos DB aquí, necesitamos pasarla o usar otro método
    # Por simplicidad, asumimos que el token es válido y tenant_id está en payload
    tenant_id = payload.get("tenant_id")
    if not tenant_id:
        await websocket.close(code=1008, reason="Tenant not found in token")
        return

    # Verificar cuota de conexiones WebSocket (simplificado)
    # En producción, consultar base de datos
    if tenant_id not in active_connections:
        active_connections[tenant_id] = {}

    current_connections = len(active_connections[tenant_id])
    if current_connections >= 10:  # Cuota fija de 10 conexiones
        await websocket.close(code=1008, reason="WebSocket quota exceeded")
        return

    # Conectar
    await websocket.accept()

    # Almacenar conexión
    connection_id = f"{model_id}_{len(active_connections[tenant_id])}"
    active_connections[tenant_id][connection_id] = websocket

    try:
        # Obtener modelo (necesitamos DB, pero WebSocket no puede usar Depends)
        # Esto es un ejemplo simplificado
        # En producción, cargar modelo desde caché o servicio

        # Loop principal: recibir filas y enviar predicciones
        while True:
            data = await websocket.receive_text()
            row_data = json.loads(data)

            # Procesar fila individual (ejemplo simplificado)
            result = await process_row_simplified(row_data, explain)

            # Enviar respuesta
            await websocket.send_json(result)

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {connection_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close(code=1011, reason=str(e))
    finally:
        # Limpiar conexión
        if (
            tenant_id in active_connections
            and connection_id in active_connections[tenant_id]
        ):
            del active_connections[tenant_id][connection_id]


async def process_row_simplified(
    row_data: Dict[str, Any], explain: bool = False
) -> Dict[str, Any]:
    """
    Procesa una fila individual y devuelve predicción.
    Versión simplificada - en producción usar lógica completa.
    """
    try:
        # Ejemplo: devolver predicción dummy
        prediction = 0.5  # Valor dummy

        result = {
            "prediction": prediction,
            "uncertainty": 0.1,
        }

        if explain:
            # Ejemplo de SHAP values dummy
            result["shap_values"] = [0.1, -0.2, 0.3]
            result["feature_names"] = list(row_data.keys())[:3]

        return result

    except Exception as e:
        logger.error(f"Error procesando fila: {e}")
        return {"error": str(e)}
