import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import os
from pathlib import Path
from typing import Optional, Dict

from app.core.config import settings


class MLFlowService:
    """
    Abstrae la conexión con MLFlow para registro y carga de modelos.
    Compatible con tracking server local (file://) o remoto (http://).
    """

    def __init__(self):
        if settings.MLFLOW_TRACKING_URI:
            tracking_uri = settings.MLFLOW_TRACKING_URI
        else:
            mlflow_dir = Path(settings.DATA_DIR) / "mlruns"
            mlflow_dir.mkdir(parents=True, exist_ok=True)
            tracking_uri = mlflow_dir.as_uri()  # file:///... correcto en Windows y Unix

        mlflow.set_tracking_uri(tracking_uri)
        self.tracking_uri = tracking_uri

    def load_model(self, run_id: str, device: torch.device) -> nn.Module:
        """
        Carga un modelo PyTorch desde MLFlow usando su run_id.
        Intenta carga estándar de MLFlow y fallback a carga manual de artefactos (.pth).
        """
        # Asegurar que el tracking URI está configurado antes de cualquier operación
        mlflow.set_tracking_uri(self.tracking_uri)

        # En el worker (dentro de Docker), 'mlflow' es el hostname del servidor
        model_uri = f"runs:/{run_id}/model"

        import logging
        logger = logging.getLogger(__name__)

        try:
            logger.info("Intentando cargar modelo pyfunc/sklearn/pytorch desde: %s", model_uri)
            try:
                # Intento 1: PyTorch
                model = mlflow.pytorch.load_model(model_uri, map_location=device)
                logger.info("Modelo cargado exitosamente con mlflow.pytorch.load_model")
                model.to(device)
                model.eval()
                return model
            except Exception as e_pt:
                logger.info("mlflow.pytorch.load_model falló (%s), intentando mlflow.sklearn", e_pt)
                import mlflow.sklearn as mlflow_sklearn
                model = mlflow_sklearn.load_model(model_uri)
                logger.info("Modelo cargado exitosamente con mlflow.sklearn.load_model")
                return model
        except Exception as e:
            logger.info("Carga estándar falló, intentando descarga manual pyTorch y reconstrucción: %s", e)

            try:
                # Recuperar metadata del run para saber qué arquitectura instanciar
                from mlflow.tracking import MlflowClient
                client = MlflowClient(tracking_uri=self.tracking_uri)
                run = client.get_run(run_id)
                tags = run.data.tags

                architecture = tags.get("architecture")
                num_classes = int(tags.get("num_classes", "2"))
                in_channels = int(tags.get("in_channels", "3")) # Asumimos 3 por defecto o MRI

                # Descargar artefactos
                mlflow.set_tracking_uri(self.tracking_uri)
                local_path = client.download_artifacts(run_id=run_id, path="model")
                logger.info("Artefactos descargados manualmente a: %s", local_path)

                # Buscar el fichero .pth, .pt, .joblib o .pkl
                file_to_load = None
                if os.path.isdir(local_path):
                    pth_files = (list(Path(local_path).glob("*.pth")) +
                               list(Path(local_path).glob("*.pt")) +
                               list(Path(local_path).glob("*.joblib")) +
                               list(Path(local_path).glob("*.pkl")))
                    if pth_files:
                        file_to_load = str(pth_files[0])
                elif local_path.endswith((".pth", ".pt", ".joblib", ".pkl")):
                    file_to_load = local_path

                if not file_to_load:
                    raise ValueError(f"No se encontró fichero de pesos (.pth/.pt/.joblib/.pkl) en {local_path}")

                if file_to_load.endswith((".joblib", ".pkl")):
                    import joblib
                    model = joblib.load(file_to_load)
                    logger.info("Modelo cargado exitosamente con joblib desde: %s", file_to_load)
                    return model

                # Cargar el objeto PyTorch (puede ser model o state_dict)
                checkpoint = torch.load(file_to_load, map_location=device, weights_only=False)

                if isinstance(checkpoint, torch.nn.Module):
                    model = checkpoint
                else:
                    # Es un state_dict o algo parecido, necesitamos instanciar la clase
                    if not architecture:
                        raise ValueError("El artefacto es un state_dict pero no hay tag 'architecture' para reconstruirlo.")

                    from app.core_ml.models.factory import ModelFactory
                    model = ModelFactory.get_model(
                        architecture=architecture,
                        in_channels=in_channels,
                        num_classes=num_classes
                    )

                    # Si es un dict anidado (común en checkpoints)
                    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
                    model.load_state_dict(state_dict)
                    logger.info("Modelo reconstruido e instanciado desde ModelFactory con state_dict")

                model.to(device)
                model.eval()
                return model

            except Exception as e2:
                logger.error("Fallo definitivo al cargar modelo: %s", e2)
                raise e2

    # ──────────────────────────────────────────────────────────────────────────
    # Model Registration
    # ──────────────────────────────────────────────────────────────────────────

    def register_pth_model(
        self,
        pth_path: str,
        model_name: str,
        tenant_id: str,
        architecture: str = "unknown",
        num_classes: int = 2,
        metrics: Optional[Dict[str, float]] = None,
        extra_tags: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Registra un fichero .pth existente en MLFlow y devuelve el run_id.

        El fichero puede ser:
          - Un modelo completo guardado con torch.save(model, path)
          - Un state_dict guardado con torch.save(model.state_dict(), path)
            En este último caso el modelo se registra como artefacto raw (.pth).

        Args:
            pth_path: Ruta absoluta al fichero .pth.
            model_name: Nombre descriptivo del modelo (run name en MLFlow).
            tenant_id: ID del tenant — determina el experimento MLFlow.
            architecture: Arquitectura del modelo (ej. 'unet', 'medsam').
            num_classes: Número de clases de segmentación.
            metrics: Métricas de entrenamiento opcionales (ej. {'dice': 0.87}).
            extra_tags: Tags adicionales para el run.

        Returns:
            str: run_id del experimento MLFlow recién creado.
        """
        experiment_name = f"tenant_{tenant_id}_models"
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run(run_name=model_name) as run:
            # Tags estructurados para facilitar filtrado en la UI
            mlflow.set_tags({
                "framework": "pytorch",
                "architecture": architecture,
                "num_classes": str(num_classes),
                "source": "pth_upload",
                "tenant_id": tenant_id,
                **(extra_tags or {}),
            })
            mlflow.log_params({
                "architecture": architecture,
                "num_classes": num_classes,
            })

            # Intentar cargar como modelo completo; si falla, guardar el .pth como artefacto raw
            try:
                pytorch_model = torch.load(pth_path, map_location="cpu", weights_only=False)
                if isinstance(pytorch_model, nn.Module):
                    mlflow.pytorch.log_model(pytorch_model, artifact_path="model")
                else:
                    # Es un state_dict — loguear el fichero raw para preservarlo
                    mlflow.log_artifact(pth_path, artifact_path="model")
                    mlflow.set_tag("model_type", "state_dict")
            except Exception:
                # Fallback: guardar el fichero raw
                mlflow.log_artifact(pth_path, artifact_path="model")
                mlflow.set_tag("model_type", "raw_pth")

            if metrics:
                mlflow.log_metrics(metrics)

            return run.info.run_id

    def log_model(
        self,
        model: nn.Module,
        model_name: str,
        tenant_id: str,
        metrics: Optional[dict] = None,
    ) -> str:
        """
        Guarda un nn.Module en MLFlow (útil para fine-tuning desde código).
        """
        experiment_name = f"tenant_{tenant_id}_models"
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run(run_name=model_name) as run:
            mlflow.pytorch.log_model(model, "model")
            if metrics:
                mlflow.log_metrics(metrics)
            return run.info.run_id

    # ──────────────────────────────────────────────────────────────────────────
    # Inference Tracking
    # ──────────────────────────────────────────────────────────────────────────

    def start_inference_run(
        self,
        model_run_id: str,
        tenant_id: str,
        method: str,
        prediction_id: str,
    ) -> mlflow.ActiveRun:
        """
        Abre un MLFlow run anidado bajo el experimento del tenant para trackear
        los métricas de una inferencia concreta.

        Returns:
            mlflow.ActiveRun: contexto del run (úsalo con `with`).
        """
        experiment_name = f"tenant_{tenant_id}_inference"
        mlflow.set_experiment(experiment_name)
        return mlflow.start_run(
            run_name=f"inference_{prediction_id[:8]}",
            tags={
                "model_run_id": model_run_id,
                "uncertainty_method": method,
                "prediction_id": prediction_id,
                "tenant_id": tenant_id,
            },
        )

    def get_tracking_uri(self) -> str:
        return self.tracking_uri
