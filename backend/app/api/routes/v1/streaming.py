"""
Endpoint WebSocket para predicciones en tiempo real.
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
import json
import logging
import os
from typing import Dict, Any, List

from app.database import SessionLocal
from app.models.ml_model import MLModel
from app.models.user import User
from app.core.security import decode_token
import torch

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
    logger.info(f"WebSocket connection attempt for model {model_id}")
    logger.info(f"Token received: {token[:10] if token else 'None'}...")
    # Validar JWT
    try:
        payload = decode_token(token)
        if not payload:
            logger.warning(f"Invalid token for model {model_id}")
            await websocket.close(code=1008, reason="Invalid token")
            return
    except Exception as e:
        logger.error(f"Error decoding token: {e}")
        await websocket.close(code=1008, reason="Token decoding error")
        return

    user_id = payload.get("sub")
    # No tenemos DB inyectada por Depends, usamos SessionLocal temporalmente si falta tenant_id
    tenant_id = payload.get("tenant_id")

    if not tenant_id and user_id:


        db = SessionLocal()
        try:
            user = db.query(User).filter(User.id == user_id).first()
            if user:
                tenant_id = user.tenant_id
        except Exception as e:
            logger.error(f"Error querying user for tenant_id: {e}")
        finally:
            db.close()

    if not tenant_id:
        await websocket.close(code=1008, reason="Tenant not found in token or db")
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
        db = SessionLocal()

        try:
            ml_model = (
                db.query(MLModel)
                .filter(MLModel.id == model_id, MLModel.tenant_id == tenant_id)
                .first()
            )
            if not ml_model:
                logger.warning(f"Modelo {model_id} no encontrado para tenant {tenant_id}")
                await websocket.close(
                    code=1008, reason=f"Modelo {model_id} no encontrado"
                )
                return

            # Copiamos los campos necesarios antes de cerrar la DB
            is_torchscript = ml_model.is_torchscript
            torchscript_path = ml_model.torchscript_path
            mlflow_run_id = ml_model.mlflow_run_id
            metadata = ml_model.metrics_metadata or {}
            preprocessing_pipeline_path = ml_model.preprocessing_pipeline_path

            # Cargar modelo real
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Usando dispositivo: {device} para modelo {model_id}")

            if is_torchscript and torchscript_path:
                logger.info(f"Cargando TorchScript model desde {torchscript_path}...")
                if not os.path.exists(torchscript_path):
                    logger.error(f"Fichero TorchScript no encontrado: {torchscript_path}")
                    await websocket.close(code=1008, reason="Fichero de modelo no encontrado en disco")
                    return
                model = torch.jit.load(torchscript_path, map_location=device)
            else:
                logger.info(f"Cargando MLFlow model para run {mlflow_run_id}...")
                from app.services.mlflow_service import MLFlowService

                mlflow_svc = MLFlowService()
                model = mlflow_svc.load_model(mlflow_run_id, device=device)

            if model is None:
                logger.error(f"No se pudo cargar el modelo {model_id}")
                await websocket.close(code=1008, reason="Error cargando modelo: model is None")
                return

            if hasattr(model, "to"):
                model.to(device)
            if hasattr(model, "eval"):
                model.eval()

            # Obtener nombres de features
            feature_names = metadata.get("feature_names", [])
            if not feature_names:
                logger.warning(
                    f"No se encontraron nombres de features para modelo {model_id}"
                )

            # Obtener pipeline de preprocesamiento si existe
            preprocessing_pipeline = None
            if preprocessing_pipeline_path:
                try:
                    from app.core_ml.preprocessing import load_pipeline
                    logger.info(f"Cargando pipeline desde {preprocessing_pipeline_path}...")
                    preprocessing_pipeline = load_pipeline(preprocessing_pipeline_path)
                except Exception as e:
                    logger.error(f"Error cargando pipeline de preprocesamiento: {e}")

        except Exception as e:
            logger.error(f"Error durante el setup del modelo: {e}", exc_info=True)
            await websocket.close(code=1011, reason=f"Error cargando modelo: {str(e)}")
            return

        # Cargar background para SHAP si se solicita inicialmente
        background_data = None
        if explain:
            background_data = await _load_background_data(db, metadata, preprocessing_pipeline, feature_names)

        db.close()

        # Loop principal: recibir filas y enviar predicciones
        while True:
            data = await websocket.receive_text()
            row_data = json.loads(data)

            # El cliente puede pedir explicaciones en cada mensaje individualmente
            msg_explain = row_data.get("explain", explain)

            # Lazy loading de background if requested but missing
            if msg_explain and background_data is None:
                logger.info("SHAP: Explicación solicitada pero background no cargado. Re-intentando carga...")
                # Re-abrimos sesión para la carga lazy
                async_db = SessionLocal()
                try:
                    background_data = await _load_background_data(async_db, metadata, preprocessing_pipeline, feature_names)
                finally:
                    async_db.close()

            # Procesar fila individual con el modelo real
            result = await process_row_real(
                row_data,
                model,
                feature_names,
                preprocessing_pipeline,
                msg_explain,
                background_data,
                task_type=metadata.get("task_type", "classification")
            )

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


async def _load_background_data(db, metadata, preprocessing_pipeline, feature_names):
    """Auxiliar para cargar background data desde el dataset original."""
    try:
        dataset_id = metadata.get("dataset_id")
        if not dataset_id:
            logger.warning("SHAP: No hay dataset_id asociado al modelo. Se usará la fila actual como background (¡No recomendado!).")
            return None

        from app.models.dataset import Dataset
        from app.services.storage_service import get_storage
        from io import BytesIO
        import pandas as pd

        dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
        if not dataset:
            logger.error(f"SHAP: No se encontró el dataset {dataset_id} en la base de datos.")
            return None
        elif not dataset.file_path:
            logger.error(f"SHAP: El dataset {dataset_id} no tiene file_path.")
            return None

        storage = get_storage()
        logger.info(f"SHAP: Intentando descargar background desde {dataset.file_path}...")
        try:
            storage_key = dataset.file_path

            logger.info(f"SHAP: Resolviendo key de storage: {storage_key}")
            data_bytes = storage.download(storage_key)
            file_type = dataset.file_type or "csv"

            if file_type == "csv":
                df_background = pd.read_csv(BytesIO(data_bytes))
            elif file_type == "parquet":
                df_background = pd.read_parquet(BytesIO(data_bytes))
            elif file_type == "xlsx":
                df_background = pd.read_excel(BytesIO(data_bytes))
            else:
                logger.warning(f"SHAP: Formato '{file_type}' no soportado directamente. Intentando con CSV.")
                df_background = pd.read_csv(BytesIO(data_bytes))

            logger.info(f"SHAP: Dataset cargado ({file_type}). Filas totales: {len(df_background)}")

            # Tomar una muestra (100 filas suelen bastar para KernelExplainer)
            sample_size = min(100, len(df_background))
            background_data = df_background.sample(n=sample_size, random_state=42)

            # Preprocesar background si hay pipeline Y el dataset NO parece estar preprocesado ya
            # Si el dataset tiene su propio pipeline_path, asumimos que el CSV ya está en formato post-pipeline
            if preprocessing_pipeline:
                if dataset.pipeline_path:
                    logger.info(f"SHAP: El dataset {dataset_id} ya parece estar preprocesado (pipeline_path={dataset.pipeline_path}). Saltando apply_pipeline.")
                else:
                    from app.core_ml.preprocessing import apply_pipeline
                    logger.info("SHAP: Aplicando pipeline de preprocesamiento al background (Dataset crudo)...")
                    background_data, _ = apply_pipeline(preprocessing_pipeline, background_data, fit=False)

            # Alinear features con lo que espera el modelo
            if feature_names:
                logger.info(f"SHAP: Alineando {len(feature_names)} features. Columnas actuales del background: {background_data.columns.tolist()[:5]}...")
                for col in feature_names:
                    if col not in background_data.columns:
                        # logger.warning(f"SHAP: Feature '{col}' no encontrada en background. Se inicializa a 0.")
                        background_data[col] = 0.0
                background_data = background_data[feature_names]

            logger.info(f"SHAP: Background data listo para usar ({len(background_data)} filas). Mean prediction on background would be a good exp_val.")
            return background_data
        except Exception as dl_err:
            logger.error(f"SHAP: Error crítico descargando/procesando background: {dl_err}", exc_info=True)
            return None
    except Exception as b_err:
        logger.error(f"SHAP: Error inesperado en el bloque de background: {b_err}", exc_info=True)
        return None


async def process_row_real(
    row_data: Dict[str, Any],
    model: Any,
    feature_names: List[str],
    preprocessing_pipeline: Any,
    explain: bool = False,
    background_data: Any = None,
    task_type: str = "classification",
) -> Dict[str, Any]:
    """
    Procesa una fila individual usando el modelo real y devuelve predicción con SHAP.
    """
    try:
        import pandas as pd
        import numpy as np
        from app.core_ml.factory import PredictionFactory, UncertaintyMethod
        from app.core_ml.preprocessing import apply_pipeline

        # Convertir datos a DataFrame
        df = pd.DataFrame([row_data])

        # Aplicar pipeline de preprocesamiento si existe
        if preprocessing_pipeline:
            df, _ = apply_pipeline(preprocessing_pipeline, df, fit=False)

        # Asegurar que tenemos todas las features necesarias (especialmente si no hay pipeline)
        # o para alinear con lo que espera el modelo post-procesamiento.
        if feature_names:
            for col in feature_names:
                if col not in df.columns:
                    df[col] = 0.0
            df = df[feature_names]

        # Convertir a numpy para inferencia
        input_data = df.to_numpy().astype(np.float32)

        # Obtener dispositivo
        device = (
            next(model.parameters()).device
            if hasattr(model, "parameters")
            else torch.device("cpu")
        )

        # Crear estimador para predicción
        # Usamos NONE como método por defecto para streaming ligero, o el que corresponda
        unc_method = UncertaintyMethod.NONE
        estimator = PredictionFactory.get_estimator(
            method=unc_method,
            model=model,
            device=device,
            mc_samples=5,
            tta_samples=5,
        )

        # Realizar predicción con incertidumbre
        result_dict = estimator.estimate_uncertainty(input_data)

        result = {
            "prediction": float(result_dict["prediction"].item())
            if hasattr(result_dict["prediction"], "item")
            else float(result_dict["prediction"]),
            "uncertainty": float(result_dict["uncertainty"].item())
            if hasattr(result_dict["uncertainty"], "item")
            else float(result_dict["uncertainty"]),
        }

        # Calcular SHAP values si se solicita
        if explain:
            try:
                from app.core_ml.explainability import get_shap_values
                n_features = len(feature_names)
                logger.info(f"Calculando SHAP para {n_features} features. Tarea: {task_type}. Background: {'Cargado' if background_data is not None else 'Ninguno (usará data)'}")

                if background_data is not None:
                    logger.info(f"Dimensiones de background_data: {getattr(background_data, 'shape', 'desconocido')}")

                # Calcular SHAP values para todos los features
                shap_result = get_shap_values(
                    model, df, feature_names,
                    background=background_data,
                    task_type=task_type
                )

                # SHAP values para la primera (y única) fila
                # get_shap_values puede devolver diversas estructuras según el explainer y versión
                raw_shap = shap_result["shap_values"]
                shap_values = []

                try:
                    # Estructura A: [clase][fila][feature] -> len(raw_shap) = n_clases, len(raw__shap[0]) = 1
                    if (isinstance(raw_shap, list) and len(raw_shap) > 0 and
                        isinstance(raw_shap[0], list) and len(raw_shap[0]) == 1 and
                        isinstance(raw_shap[0][0], list) and len(raw_shap[0][0]) == n_features):
                        class_idx = 1 if len(raw_shap) == 2 else 0
                        shap_values = raw_shap[class_idx][0]

                    # Estructura B: [fila][feature][clase] -> len(raw_shap) = 1, len(raw_shap[0]) = n_features
                    elif (isinstance(raw_shap, list) and len(raw_shap) == 1 and
                          isinstance(raw_shap[0], list) and len(raw_shap[0]) == n_features):
                        if isinstance(raw_shap[0][0], list):
                            # Cada feature tiene una lista de contribuciones por clase
                            class_idx = 1 if len(raw_shap[0][0]) == 2 else 0
                            shap_values = [f[class_idx] for f in raw_shap[0]]
                        else:
                            # Caso simple [fila][feature]
                            shap_values = raw_shap[0]

                    # Estructura C: [fila][feature] directa (Común en regresión o single-output)
                    elif isinstance(raw_shap, list) and len(raw_shap) > 0 and isinstance(raw_shap[0], list) and len(raw_shap[0]) == n_features:
                         shap_values = raw_shap[0]

                    else:
                        # Fallback: intentar tomar el primer elemento si es lista, o el objeto directo
                        shap_values = raw_shap[0] if (isinstance(raw_shap, list) and len(raw_shap) > 0) else raw_shap

                except Exception as unpack_err:
                    logger.warning(f"Error desempaquetando SHAP: {unpack_err}. Usando raw_shap.")
                    shap_values = raw_shap

                result["shap_values"] = shap_values
                result["feature_names"] = feature_names

                # Asegurar que expected_value es un escalar para el frontend
                exp_val = shap_result["expected_value"]
                if task_type == "classification" and isinstance(exp_val, (list, np.ndarray)) and len(exp_val) > 1:
                    # Si es lista de esperados por clase, elegir la misma clase que en SHAP
                    class_idx = 1 if len(exp_val) == 2 else 0
                    result["expected_value"] = float(exp_val[class_idx])
                elif isinstance(exp_val, (list, np.ndarray)) and len(exp_val) >= 1:
                    result["expected_value"] = float(exp_val[0])
                else:
                    result["expected_value"] = float(exp_val)

                logger.info(f"SHAP completado. Exp_val={result['expected_value']}, Out_val={result.get('prediction')}")

            except Exception as shap_error:
                logger.error(f"Error calculando SHAP values: {shap_error}")
                # Devolver valores vacíos si SHAP falla, pero manteniendo la predicción
                result["shap_values"] = []
                result["feature_names"] = feature_names

        return result

    except Exception as e:
        logger.error(f"Error procesando fila: {e}")
        return {"error": str(e)}
