"""
Servicio de entrenamiento para modelos Scikit-learn y PyTorch.
Gestiona el ciclo completo: instanciación, train/test split o
cross-validation, entrenamiento con MLFlow autologging, evaluación y persistencia.
"""

import importlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)

from app.core.config import settings
from app.core_ml.hyperparams import get_algorithm_info, get_default_hyperparams
from app.services.training_utils import prepare_features

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Sklearn Trainer
# ═══════════════════════════════════════════════════════════════════════════════


class SklearnTrainer:
    """
    Entrena modelos sklearn con MLFlow autologging completo.

    Flujo:
        1. Cargar datos y separar features/target
        2. Train/test split
        3. Instanciar modelo con hiperparámetros
        4. mlflow.sklearn.autolog() + model.fit()
        5. Evaluar en test set
        6. Guardar modelo con joblib
        7. Devolver métricas y run_id de MLFlow
    """

    def __init__(self, tenant_id: str, mlflow_tracking_uri: Optional[str] = None):
        self.tenant_id = tenant_id
        if mlflow_tracking_uri:
            mlflow.set_tracking_uri(mlflow_tracking_uri)
        elif settings.MLFLOW_TRACKING_URI:
            mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)

    def _import_model_class(self, class_path: str) -> type:
        """Importa dinámicamente una clase sklearn dado su path completo."""
        module_path, class_name = class_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)

    def _instantiate_model(
        self,
        algorithm: str,
        task_type: str,
        hyperparams: Dict[str, Any],
    ) -> Any:
        """Instancia el modelo sklearn con los hiperparámetros dados."""
        algo_info = get_algorithm_info(algorithm)

        if task_type == "regression" and "sklearn_regressor" in algo_info:
            class_path = algo_info["sklearn_regressor"]
        else:
            class_path = algo_info["sklearn_class"]

        model_class = self._import_model_class(class_path)

        # Merge defaults con los proporcionados por el usuario
        defaults = get_default_hyperparams(algorithm)
        final_params = {**defaults, **hyperparams}

        # Filtrar parámetros que el constructor no acepta
        import inspect

        valid_params = inspect.signature(model_class.__init__).parameters
        filtered = {k: v for k, v in final_params.items() if k in valid_params}

        # SVM: habilitar probabilidades para incertidumbre
        if algorithm == "svm" and task_type == "classification":
            filtered["probability"] = True

        logger.info("Instanciando %s con params: %s", class_path, filtered)
        return model_class(**filtered)

    def train(
        self,
        df: pd.DataFrame,
        target_column: str,
        algorithm: str,
        task_type: str = "classification",
        hyperparams: Optional[Dict[str, Any]] = None,
        validation_config: Optional[Dict[str, Any]] = None,
        model_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Ejecuta el pipeline completo de entrenamiento sklearn.

        Args:
            validation_config: {
                "strategy": "holdout" | "cross_validation",
                "test_size": 0.2,          # for holdout
                "n_folds": 5,              # for CV
                "shuffle": True,
                "random_state": 42,
            }

        Returns:
            Dict con: model, metrics, mlflow_run_id, model_path, feature_names, cv_results (if CV)
        """
        hyperparams = hyperparams or {}
        vc = validation_config or {}
        strategy = vc.get("strategy", "holdout")
        test_size = vc.get("test_size", 0.2)
        n_folds = vc.get("n_folds", 5)
        shuffle = vc.get("shuffle", True)
        random_state = vc.get("random_state", 42)

        if target_column not in df.columns:
            raise ValueError(f"Columna objetivo '{target_column}' no encontrada.")

        X = df.drop(columns=[target_column])
        y = df[target_column]
        X = prepare_features(X)

        algo_info = get_algorithm_info(algorithm)
        experiment_name = f"tenant_{self.tenant_id}_training"
        mlflow.set_experiment(experiment_name)
        mlflow.sklearn.autolog(log_models=True, log_datasets=False)

        if strategy == "cross_validation":
            return self._train_cv(
                X,
                y,
                algorithm,
                task_type,
                hyperparams,
                algo_info,
                n_folds,
                shuffle,
                random_state,
                target_column,
                model_name,
            )
        else:
            return self._train_holdout(
                X,
                y,
                algorithm,
                task_type,
                hyperparams,
                algo_info,
                test_size,
                shuffle,
                random_state,
                target_column,
                model_name,
            )

    def _train_holdout(
        self,
        X,
        y,
        algorithm,
        task_type,
        hyperparams,
        algo_info,
        test_size,
        shuffle,
        random_state,
        target_column,
        model_name=None,
    ) -> Dict[str, Any]:
        """Entrenamiento con holdout train/test split."""
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            shuffle=shuffle,
            stratify=y
            if task_type == "classification" and shuffle and len(y.unique()) > 1
            else None,
        )

        logger.info(
            "Holdout split: train=%d, test=%d, features=%d",
            len(X_train),
            len(X_test),
            X_train.shape[1],
        )

        model = self._instantiate_model(algorithm, task_type, hyperparams)

        run_name = model_name or algo_info["display_name"]
        with mlflow.start_run(run_name=run_name) as run:
            mlflow.set_tags(
                {
                    "framework": "sklearn",
                    "algorithm": algorithm,
                    "task_type": task_type,
                    "target_column": target_column,
                    "tenant_id": self.tenant_id,
                    "validation_strategy": "holdout",
                    "test_size": str(test_size),
                    "n_features": str(X_train.shape[1]),
                    "n_train_samples": str(len(X_train)),
                    "n_test_samples": str(len(X_test)),
                }
            )

            model.fit(X_train, y_train)
            metrics = self._evaluate(model, X_test, y_test, task_type)
            mlflow.log_metrics({f"test_{k}": v for k, v in metrics.items()})

            model_path = self._save_model(model, algorithm, run.info.run_id)
            mlflow_run_id = run.info.run_id

        mlflow.sklearn.autolog(disable=True)

        logger.info(
            "Holdout training completo: run_id=%s, metrics=%s", mlflow_run_id, metrics
        )

        return {
            "model": model,
            "metrics": metrics,
            "mlflow_run_id": mlflow_run_id,
            "model_path": str(model_path),
            "feature_names": X.columns.tolist(),
            "validation_strategy": "holdout",
            "X_test": X_test,
            "y_test": y_test,
        }

    def _train_cv(
        self,
        X,
        y,
        algorithm,
        task_type,
        hyperparams,
        algo_info,
        n_folds,
        shuffle,
        random_state,
        target_column,
        model_name=None,
    ) -> Dict[str, Any]:
        """Entrenamiento con k-fold cross-validation."""
        if task_type == "classification" and len(y.unique()) > 1:
            cv = StratifiedKFold(
                n_splits=n_folds, shuffle=shuffle, random_state=random_state
            )
        else:
            cv = KFold(n_splits=n_folds, shuffle=shuffle, random_state=random_state)

        model = self._instantiate_model(algorithm, task_type, hyperparams)

        # Scoring metrics per task type
        if task_type == "classification":
            scoring = [
                "accuracy",
                "f1_weighted",
                "precision_weighted",
                "recall_weighted",
            ]
        else:
            scoring = ["neg_mean_squared_error", "neg_mean_absolute_error", "r2"]

        logger.info(
            "Cross-validation: %d folds, features=%d, samples=%d",
            n_folds,
            X.shape[1],
            len(X),
        )

        run_name = (
            f"{model_name} (CV-{n_folds})"
            if model_name
            else f"{algo_info['display_name']} (CV-{n_folds})"
        )
        with mlflow.start_run(run_name=run_name) as run:
            mlflow.set_tags(
                {
                    "framework": "sklearn",
                    "algorithm": algorithm,
                    "task_type": task_type,
                    "target_column": target_column,
                    "tenant_id": self.tenant_id,
                    "validation_strategy": "cross_validation",
                    "n_folds": str(n_folds),
                    "n_features": str(X.shape[1]),
                    "n_samples": str(len(X)),
                }
            )

            # Run cross-validation
            cv_results = cross_validate(
                model,
                X,
                y,
                cv=cv,
                scoring=scoring,
                return_train_score=True,
                n_jobs=-1,
            )

            # Compute per-fold and mean metrics
            metrics = {}
            cv_detail = {}
            if task_type == "classification":
                fold_metrics_map = {
                    "accuracy": "test_accuracy",
                    "f1": "test_f1_weighted",
                    "precision": "test_precision_weighted",
                    "recall": "test_recall_weighted",
                }
            else:
                fold_metrics_map = {
                    "mse": "test_neg_mean_squared_error",
                    "mae": "test_neg_mean_absolute_error",
                    "r2": "test_r2",
                }

            for display_name, cv_key in fold_metrics_map.items():
                values = cv_results[cv_key]
                if "neg_" in cv_key:
                    values = -values  # Convert negated metrics
                metrics[f"cv_mean_{display_name}"] = float(np.mean(values))
                metrics[f"cv_std_{display_name}"] = float(np.std(values))
                cv_detail[display_name] = {
                    "per_fold": [float(v) for v in values],
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                }

            # Log mean metrics to MLFlow
            mlflow.log_metrics(metrics)

            # Log per-fold as params (for traceability)
            for display_name, detail in cv_detail.items():
                for fold_idx, val in enumerate(detail["per_fold"]):
                    mlflow.log_metric(f"fold_{fold_idx}_{display_name}", val)

            # Final fit on all data for the production model
            final_model = self._instantiate_model(algorithm, task_type, hyperparams)
            final_model.fit(X, y)

            model_path = self._save_model(final_model, algorithm, run.info.run_id)
            mlflow_run_id = run.info.run_id

        mlflow.sklearn.autolog(disable=True)

        logger.info(
            "CV training completo: run_id=%s, metrics=%s", mlflow_run_id, metrics
        )

        return {
            "model": final_model,
            "metrics": metrics,
            "mlflow_run_id": mlflow_run_id,
            "model_path": str(model_path),
            "feature_names": X.columns.tolist(),
            "validation_strategy": "cross_validation",
            "n_folds": n_folds,
            "cv_detail": cv_detail,
        }

    def _save_model(self, model, algorithm: str, run_id: str) -> Path:
        """Guarda el modelo con joblib y lo registra como artefacto MLFlow."""
        models_dir = Path(settings.DATA_DIR) / "tenants" / self.tenant_id / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        model_filename = f"{algorithm}_{run_id[:8]}.joblib"
        model_path = models_dir / model_filename
        joblib.dump(model, model_path)
        mlflow.log_artifact(str(model_path), artifact_path="model")
        return model_path

    def _evaluate(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        task_type: str,
    ) -> Dict[str, float]:
        """Calcula métricas de evaluación sklearn."""
        y_pred = model.predict(X_test)

        if task_type == "classification":
            is_binary = len(np.unique(y_test)) <= 2
            avg = "binary" if is_binary else "weighted"
            return {
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "f1": float(f1_score(y_test, y_pred, average=avg, zero_division=0)),
                "precision": float(
                    precision_score(y_test, y_pred, average=avg, zero_division=0)
                ),
                "recall": float(
                    recall_score(y_test, y_pred, average=avg, zero_division=0)
                ),
            }
        else:
            return {
                "mse": float(mean_squared_error(y_test, y_pred)),
                "mae": float(mean_absolute_error(y_test, y_pred)),
                "r2": float(r2_score(y_test, y_pred)),
            }


# ═══════════════════════════════════════════════════════════════════════════════
# PyTorch Trainer
# ═══════════════════════════════════════════════════════════════════════════════


class PyTorchTrainer:
    """
    Entrena modelos PyTorch (nn.Module) definidos por el usuario.
    Soporta datos tabulares (clasificación/regresión) y opcionalmente imágenes.

    Flujo:
        1. Preparar Dataset/DataLoader a partir del DataFrame
        2. Instanciar la arquitectura de red (MLP, custom, o registrada)
        3. Configurar optimizer + loss + scheduler
        4. mlflow.pytorch.autolog() + training loop (epochs)
        5. Evaluar en test set
        6. Guardar modelo como .pt (state_dict) + TorchScript si posible
        7. Devolver métricas y run_id de MLFlow
    """

    def __init__(self, tenant_id: str, mlflow_tracking_uri: Optional[str] = None):
        import torch

        self.tenant_id = tenant_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if mlflow_tracking_uri:
            mlflow.set_tracking_uri(mlflow_tracking_uri)
        elif settings.MLFLOW_TRACKING_URI:
            mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)

    def _build_mlp(
        self,
        input_dim: int,
        output_dim: int,
        task_type: str,
        hyperparams: Dict[str, Any],
    ):
        """
        Construye un MLP (Multi-Layer Perceptron) configurable.
        El usuario puede definir hidden_layers, dropout, y activación.
        """
        import torch.nn as nn

        hidden_layers = hyperparams.get("hidden_layers", [128, 64])
        dropout = hyperparams.get("dropout", 0.2)
        activation = hyperparams.get("activation", "relu")

        act_fn = {
            "relu": nn.ReLU,
            "leaky_relu": nn.LeakyReLU,
            "elu": nn.ELU,
            "selu": nn.SELU,
            "tanh": nn.Tanh,
            "gelu": nn.GELU,
        }.get(activation, nn.ReLU)

        layers = []
        prev_dim = input_dim
        for h in hidden_layers:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(act_fn())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))

        model = nn.Sequential(*layers)
        return model

    def _get_model(
        self,
        algorithm: str,
        input_dim: int,
        output_dim: int,
        task_type: str,
        hyperparams: Dict[str, Any],
    ):
        """
        Obtiene el modelo PyTorch.
        - 'mlp': construye un MLP configurable
        - Cualquier nombre registrado en ModelFactory (e.g. 'unet')
        """

        if algorithm == "mlp":
            return self._build_mlp(input_dim, output_dim, task_type, hyperparams)

        # Intentar cargar de la factory de modelos registrados
        from app.core_ml.models.factory import ModelFactory

        try:
            model = ModelFactory.get_model(
                algorithm,
                in_channels=hyperparams.get("in_channels", input_dim),
                num_classes=output_dim,
                **{
                    k: v
                    for k, v in hyperparams.items()
                    if k
                    not in (
                        "in_channels",
                        "num_classes",
                        "epochs",
                        "learning_rate",
                        "batch_size",
                        "optimizer",
                        "scheduler",
                        "weight_decay",
                        "hidden_layers",
                        "dropout",
                        "activation",
                    )
                },
            )
            return model
        except ValueError:
            raise ValueError(
                f"Arquitectura PyTorch '{algorithm}' no encontrada. "
                f"Usa 'mlp' para un perceptrón multicapa o registra la arquitectura en ModelFactory."
            )

    def train(
        self,
        df: pd.DataFrame,
        target_column: str,
        algorithm: str,
        task_type: str = "classification",
        hyperparams: Optional[Dict[str, Any]] = None,
        validation_config: Optional[Dict[str, Any]] = None,
        model_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Pipeline de entrenamiento PyTorch completo para datos tabulares.
        Soporta holdout y cross-validation.
        """
        import mlflow.pytorch
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        hyperparams = hyperparams or {}
        vc = validation_config or {}
        strategy = vc.get("strategy", "holdout")
        test_size = vc.get("test_size", 0.2)
        n_folds = vc.get("n_folds", 5)
        shuffle = vc.get("shuffle", True)
        random_state = vc.get("random_state", 42)

        if target_column not in df.columns:
            raise ValueError(f"Columna objetivo '{target_column}' no encontrada.")

        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Preparar features numéricas
        X = prepare_features(X)

        # Encoding de labels para clasificación
        label_map = None
        if task_type == "classification":
            unique_labels = sorted(y.unique())
            label_map = {label: idx for idx, label in enumerate(unique_labels)}
            y = y.map(label_map)
            output_dim = len(unique_labels)
        else:
            output_dim = 1

        # Split (holdout — for CV, this will be handled per-fold inside _train_pytorch_fold)
        X_np = X.values.astype(np.float32)
        y_np = y.values.astype(np.float32 if task_type == "regression" else np.int64)

        if strategy == "cross_validation":
            return self._train_pytorch_cv(
                X,
                X_np,
                y_np,
                algorithm,
                task_type,
                hyperparams,
                label_map,
                output_dim,
                n_folds,
                shuffle,
                random_state,
                target_column,
                model_name,
            )

        X_train, X_test, y_train, y_test = train_test_split(
            X_np,
            y_np,
            test_size=test_size,
            random_state=random_state,
            shuffle=shuffle,
            stratify=y_np if task_type == "classification" else None,
        )

        input_dim = X_train.shape[1]
        logger.info(
            "PyTorch Split: train=%d, test=%d, input_dim=%d, output_dim=%d",
            len(X_train),
            len(X_test),
            input_dim,
            output_dim,
        )

        # DataLoaders
        batch_size = hyperparams.get("batch_size", 64)
        train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
        test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        # Modelo
        model = self._get_model(
            algorithm, input_dim, output_dim, task_type, hyperparams
        )
        model = model.to(self.device)

        # Loss
        if task_type == "classification":
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.MSELoss()

        # Optimizer
        lr = hyperparams.get("learning_rate", 0.001)
        weight_decay = hyperparams.get("weight_decay", 0.0)
        opt_name = hyperparams.get("optimizer", "adam")
        if opt_name == "sgd":
            optimizer = torch.optim.SGD(
                model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9
            )
        elif opt_name == "adamw":
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=lr, weight_decay=weight_decay
            )
        else:
            optimizer = torch.optim.Adam(
                model.parameters(), lr=lr, weight_decay=weight_decay
            )

        # Scheduler
        sched_name = hyperparams.get("scheduler", "none")
        scheduler = None
        epochs = hyperparams.get("epochs", 50)
        if sched_name == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs
            )
        elif sched_name == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=max(epochs // 3, 1), gamma=0.1
            )
        elif sched_name == "reduce_on_plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, patience=5, factor=0.5
            )

        # MLFlow
        experiment_name = f"tenant_{self.tenant_id}_training"
        mlflow.set_experiment(experiment_name)
        mlflow.pytorch.autolog(log_models=True)

        algo_display = algorithm.upper() if algorithm != "mlp" else "MLP"
        run_name = model_name or f"PyTorch {algo_display}"

        with mlflow.start_run(run_name=run_name) as run:
            mlflow.set_tags(
                {
                    "framework": "pytorch",
                    "algorithm": algorithm,
                    "task_type": task_type,
                    "target_column": target_column,
                    "tenant_id": self.tenant_id,
                    "n_features": str(input_dim),
                    "n_train_samples": str(len(X_train)),
                    "n_test_samples": str(len(X_test)),
                    "validation_strategy": "holdout",
                }
            )
            mlflow.log_params(
                {
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "learning_rate": lr,
                    "optimizer": opt_name,
                    "scheduler": sched_name,
                    "weight_decay": weight_decay,
                    "architecture": algorithm,
                }
            )
            if algorithm == "mlp":
                mlflow.log_params(
                    {
                        "hidden_layers": str(
                            hyperparams.get("hidden_layers", [128, 64])
                        ),
                        "dropout": hyperparams.get("dropout", 0.2),
                        "activation": hyperparams.get("activation", "relu"),
                    }
                )

            # ── Training loop ────────────────────────────────────────────────
            best_loss = float("inf")
            for epoch in range(epochs):
                model.train()
                epoch_loss = 0.0
                for X_batch, y_batch in train_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)

                    optimizer.zero_grad()
                    outputs = model(X_batch)

                    if task_type == "regression":
                        outputs = outputs.squeeze(-1)
                    loss = criterion(outputs, y_batch)
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item() * X_batch.size(0)

                epoch_loss /= len(train_loader.dataset)

                # Scheduler step
                if scheduler is not None:
                    if sched_name == "reduce_on_plateau":
                        scheduler.step(epoch_loss)
                    else:
                        scheduler.step()

                # Log cada 10% de epochs, o always para primeras y últimas
                if epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1:
                    mlflow.log_metrics(
                        {"train_loss": epoch_loss, "epoch": epoch}, step=epoch
                    )
                    logger.info(
                        "Epoch %d/%d — loss: %.6f", epoch + 1, epochs, epoch_loss
                    )

                if epoch_loss < best_loss:
                    best_loss = epoch_loss

            # ── Evaluate on test ─────────────────────────────────────────────
            metrics = self._evaluate_pytorch(model, test_loader, task_type, criterion)
            mlflow.log_metrics({f"test_{k}": v for k, v in metrics.items()})

            # ── Save model ───────────────────────────────────────────────────
            models_dir = Path(settings.DATA_DIR) / "tenants" / self.tenant_id / "models"
            models_dir.mkdir(parents=True, exist_ok=True)
            model_filename = f"pytorch_{algorithm}_{run.info.run_id[:8]}.pt"
            model_path = models_dir / model_filename
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "algorithm": algorithm,
                    "input_dim": input_dim,
                    "output_dim": output_dim,
                    "task_type": task_type,
                    "hyperparams": hyperparams,
                    "label_map": label_map,
                    "feature_names": X.columns.tolist(),
                },
                model_path,
            )

            mlflow.log_artifact(str(model_path), artifact_path="model")

            # Intentar guardar como TorchScript
            try:
                model.eval()
                sample = torch.randn(1, input_dim).to(self.device)
                scripted = torch.jit.trace(model, sample)
                ts_path = (
                    models_dir
                    / f"pytorch_{algorithm}_{run.info.run_id[:8]}_scripted.pt"
                )
                scripted.save(str(ts_path))
                mlflow.log_artifact(str(ts_path), artifact_path="model")
                logger.info("TorchScript model saved: %s", ts_path)
            except Exception as e:
                logger.warning("TorchScript export failed (non-critical): %s", e)

            mlflow_run_id = run.info.run_id

        mlflow.pytorch.autolog(disable=True)

        logger.info(
            "PyTorch training completo: algorithm=%s, run_id=%s, metrics=%s",
            algorithm,
            mlflow_run_id,
            metrics,
        )

        return {
            "model": model,
            "metrics": metrics,
            "mlflow_run_id": mlflow_run_id,
            "model_path": str(model_path),
            "feature_names": X.columns.tolist(),
            "validation_strategy": "holdout",
            "X_test": X_test,
            "y_test": y_test,
        }

    def _train_pytorch_cv(
        self,
        X_df,
        X_np,
        y_np,
        algorithm,
        task_type,
        hyperparams,
        label_map,
        output_dim,
        n_folds,
        shuffle,
        random_state,
        target_column,
        model_name=None,
    ) -> Dict[str, Any]:
        """k-fold cross-validation para PyTorch."""
        import mlflow.pytorch
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        input_dim = X_np.shape[1]
        batch_size = hyperparams.get("batch_size", 64)
        epochs = hyperparams.get("epochs", 50)

        if task_type == "classification":
            cv = StratifiedKFold(
                n_splits=n_folds, shuffle=shuffle, random_state=random_state
            )
            split_target = y_np
        else:
            cv = KFold(n_splits=n_folds, shuffle=shuffle, random_state=random_state)
            split_target = None

        all_fold_metrics: List[Dict[str, float]] = []

        experiment_name = f"tenant_{self.tenant_id}_training"
        mlflow.set_experiment(experiment_name)
        mlflow.pytorch.autolog(log_models=True)
        algo_display = algorithm.upper() if algorithm != "mlp" else "MLP"
        run_name = (
            f"{model_name} (CV-{n_folds})"
            if model_name
            else f"PyTorch {algo_display} (CV-{n_folds})"
        )

        with mlflow.start_run(run_name=run_name) as run:
            mlflow.set_tags(
                {
                    "framework": "pytorch",
                    "algorithm": algorithm,
                    "task_type": task_type,
                    "target_column": target_column,
                    "tenant_id": self.tenant_id,
                    "validation_strategy": "cross_validation",
                    "n_folds": str(n_folds),
                    "n_features": str(input_dim),
                    "n_samples": str(len(X_np)),
                }
            )

            for fold_idx, (train_idx, val_idx) in enumerate(
                cv.split(X_np, split_target)
                if split_target is not None
                else cv.split(X_np)
            ):
                logger.info("Fold %d/%d", fold_idx + 1, n_folds)

                X_tr, X_val = X_np[train_idx], X_np[val_idx]
                y_tr, y_val = y_np[train_idx], y_np[val_idx]

                train_ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
                val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
                train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
                val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

                fold_model = self._get_model(
                    algorithm, input_dim, output_dim, task_type, hyperparams
                )
                fold_model = fold_model.to(self.device)

                criterion = (
                    nn.CrossEntropyLoss()
                    if task_type == "classification"
                    else nn.MSELoss()
                )
                lr = hyperparams.get("learning_rate", 0.001)
                optimizer = torch.optim.Adam(
                    fold_model.parameters(),
                    lr=lr,
                    weight_decay=hyperparams.get("weight_decay", 0.0),
                )

                for _epoch in range(epochs):
                    fold_model.train()
                    for X_b, y_b in train_loader:
                        X_b, y_b = X_b.to(self.device), y_b.to(self.device)
                        optimizer.zero_grad()
                        out = fold_model(X_b)
                        if task_type == "regression":
                            out = out.squeeze(-1)
                        loss = criterion(out, y_b)
                        loss.backward()
                        optimizer.step()

                fold_metrics = self._evaluate_pytorch(
                    fold_model, val_loader, task_type, criterion
                )
                all_fold_metrics.append(fold_metrics)

                for k, v in fold_metrics.items():
                    mlflow.log_metric(f"fold_{fold_idx}_{k}", v)

            # Aggregate CV metrics
            metrics = {}
            cv_detail = {}
            all_keys = all_fold_metrics[0].keys()
            for k in all_keys:
                values = [fm[k] for fm in all_fold_metrics]
                metrics[f"cv_mean_{k}"] = float(np.mean(values))
                metrics[f"cv_std_{k}"] = float(np.std(values))
                cv_detail[k] = {
                    "per_fold": values,
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                }
            mlflow.log_metrics(metrics)

            # Final model trained on all data
            final_model = self._get_model(
                algorithm, input_dim, output_dim, task_type, hyperparams
            )
            final_model = final_model.to(self.device)
            criterion = (
                nn.CrossEntropyLoss() if task_type == "classification" else nn.MSELoss()
            )
            optimizer = torch.optim.Adam(
                final_model.parameters(), lr=hyperparams.get("learning_rate", 0.001)
            )
            full_ds = TensorDataset(torch.from_numpy(X_np), torch.from_numpy(y_np))
            full_loader = DataLoader(full_ds, batch_size=batch_size, shuffle=True)

            for _epoch in range(epochs):
                final_model.train()
                for X_b, y_b in full_loader:
                    X_b, y_b = X_b.to(self.device), y_b.to(self.device)
                    optimizer.zero_grad()
                    out = final_model(X_b)
                    if task_type == "regression":
                        out = out.squeeze(-1)
                    loss = criterion(out, y_b)
                    loss.backward()
                    optimizer.step()

            # Save final model
            models_dir = Path(settings.DATA_DIR) / "tenants" / self.tenant_id / "models"
            models_dir.mkdir(parents=True, exist_ok=True)
            model_filename = f"pytorch_{algorithm}_{run.info.run_id[:8]}.pt"
            model_path = models_dir / model_filename
            torch.save(
                {
                    "model_state_dict": final_model.state_dict(),
                    "algorithm": algorithm,
                    "input_dim": input_dim,
                    "output_dim": output_dim,
                    "task_type": task_type,
                    "hyperparams": hyperparams,
                    "label_map": label_map,
                    "feature_names": X_df.columns.tolist(),
                },
                model_path,
            )
            mlflow.log_artifact(str(model_path), artifact_path="model")
            mlflow_run_id = run.info.run_id

        mlflow.pytorch.autolog(disable=True)

        logger.info(
            "PyTorch CV training completo: run_id=%s, metrics=%s",
            mlflow_run_id,
            metrics,
        )

        return {
            "model": final_model,
            "metrics": metrics,
            "mlflow_run_id": mlflow_run_id,
            "model_path": str(model_path),
            "feature_names": X_df.columns.tolist(),
            "validation_strategy": "cross_validation",
            "n_folds": n_folds,
            "cv_detail": cv_detail,
        }

    def _evaluate_pytorch(
        self,
        model,
        test_loader,
        task_type: str,
        criterion,
    ) -> Dict[str, float]:
        """Evalúa modelo PyTorch en test set."""
        import torch

        model.eval()
        all_preds = []
        all_targets = []
        total_loss = 0.0

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                outputs = model(X_batch)

                if task_type == "regression":
                    outputs = outputs.squeeze(-1)
                    loss = criterion(outputs, y_batch)
                    preds = outputs.cpu().numpy()
                else:
                    loss = criterion(outputs, y_batch)
                    preds = outputs.argmax(dim=1).cpu().numpy()

                total_loss += loss.item() * X_batch.size(0)
                all_preds.extend(preds)
                all_targets.extend(y_batch.cpu().numpy())

        total_loss /= len(test_loader.dataset)
        y_pred = np.array(all_preds)
        y_true = np.array(all_targets)

        metrics: Dict[str, float] = {"loss": total_loss}

        if task_type == "classification":
            is_binary = len(np.unique(y_true)) <= 2
            avg = "binary" if is_binary else "weighted"
            metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
            metrics["f1"] = float(
                f1_score(y_true, y_pred, average=avg, zero_division=0)
            )
            metrics["precision"] = float(
                precision_score(y_true, y_pred, average=avg, zero_division=0)
            )
            metrics["recall"] = float(
                recall_score(y_true, y_pred, average=avg, zero_division=0)
            )
        else:
            metrics["mse"] = float(mean_squared_error(y_true, y_pred))
            metrics["mae"] = float(mean_absolute_error(y_true, y_pred))
            metrics["r2"] = float(r2_score(y_true, y_pred))

        return metrics
