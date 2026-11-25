"""
Servicio de entrenamiento NLP con HuggingFace Transformers + MLflow autologging.

Soporta:
  - text_classification / sentiment_analysis → AutoModelForSequenceClassification
  - token_classification (NER)               → AutoModelForTokenClassification

Flujo:
  1. Cargar datos y construir label_map (id ↔ label).
  2. Tokenizar (padding/truncation a max_seq_length).
  3. Train/val split estratificado (holdout) o k-fold cross-validation.
  4. Trainer + TrainingArguments con MLflow autolog.
  5. Métricas: sklearn (clasificación) o seqeval (NER por entidad + overall).
  6. Guardar modelo + tokenizer, exportar a ONNX y loguear artefactos en MLflow.
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import mlflow
import numpy as np
import pandas as pd
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from app.core.config import settings
from app.platform.tracker.mlflow_service import MLFlowService
from app.core_ml.hyperparams import (
    get_nlp_algorithm_info,
    get_nlp_default_hyperparams,
)
from app.core_ml.nlp.onnx_export import export_model_to_onnx

logger = logging.getLogger(__name__)


class NLPTrainer:
    """Entrena modelos NLP (HuggingFace) con MLflow + export ONNX."""

    PRETRAINED_MODELS: Dict[str, str] = {
        "bert-base": "bert-base-uncased",
        "bert-multilingual": "bert-base-multilingual-cased",
        "distilbert": "distilbert-base-uncased",
        "roberta": "roberta-base",
        "xlm-roberta": "xlm-roberta-base",
        "biobert": "dmis-lab/biobert-v1.1",
    }

    # task_type → clase de modelo Auto*
    TASK_TYPES: Dict[str, str] = {
        "text_classification": "AutoModelForSequenceClassification",
        "sentiment_analysis": "AutoModelForSequenceClassification",
        "token_classification": "AutoModelForTokenClassification",
    }

    def __init__(
        self,
        tenant_id: str,
        mlflow_tracking_uri: Optional[str] = None,
        device: Optional[str] = None,
    ):
        self.tenant_id = tenant_id
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        if mlflow_tracking_uri:
            mlflow.set_tracking_uri(mlflow_tracking_uri)
        elif settings.MLFLOW_TRACKING_URI:
            mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)

    # ──────────────────────────────────────────────────────────────────────
    # API pública
    # ──────────────────────────────────────────────────────────────────────

    def train(
        self,
        df: pd.DataFrame,
        text_column: str,
        label_column: Optional[str] = None,
        algorithm: str = "distilbert",
        task_type: str = "text_classification",
        hyperparams: Optional[Dict[str, Any]] = None,
        validation_config: Optional[Dict[str, Any]] = None,
        model_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Pipeline completo de entrenamiento NLP."""
        hp = {
            **get_nlp_default_hyperparams(algorithm),
            **(hyperparams or {}),
        }
        vc = validation_config or {}
        strategy = vc.get("strategy", "holdout")
        test_size = vc.get("test_size", 0.2)
        n_folds = vc.get("n_folds", 5)
        shuffle = vc.get("shuffle", True)
        random_state = vc.get("random_state", 42)

        if text_column not in df.columns:
            raise ValueError(f"Columna de texto '{text_column}' no encontrada.")

        if task_type == "token_classification":
            if not label_column:
                raise ValueError("token_classification requiere label_column (labels BIO).")
            if label_column not in df.columns:
                raise ValueError(f"Columna de labels '{label_column}' no encontrada.")
        elif not label_column:
            raise ValueError("Se requiere label_column para clasificación de texto.")
        elif label_column not in df.columns:
            raise ValueError(f"Columna objetivo '{label_column}' no encontrada.")

        # Filtrar filas vacías
        df = df[[text_column, label_column]].dropna()
        df = df[df[text_column].astype(str).str.strip() != ""]
        if len(df) == 0:
            raise ValueError("El dataset no contiene filas válidas de texto.")

        algo_info = get_nlp_algorithm_info(algorithm)
        hf_model = algo_info["hf_model"]

        self._setup_mlflow(model_name or algo_info["display_name"], algorithm, task_type, text_column, label_column, hp)

        if strategy == "cross_validation":
            result = self._train_cv(
                df, text_column, label_column, hf_model, task_type, hp,
                n_folds, shuffle, random_state, algorithm,
            )
        else:
            result = self._train_holdout(
                df, text_column, label_column, hf_model, task_type, hp,
                test_size, shuffle, random_state, algorithm,
            )

        result["feature_names"] = [text_column]
        result["algorithm"] = algorithm
        result["device"] = self.device
        return result

    # ──────────────────────────────────────────────────────────────────────
    # MLflow setup
    # ──────────────────────────────────────────────────────────────────────

    def _setup_mlflow(
        self, run_name: str, algorithm: str, task_type: str,
        text_column: str, label_column: Optional[str], hp: Dict[str, Any],
    ) -> None:
        MLFlowService().ensure_tenant_experiment(self.tenant_id, "nlp-training")
        mlflow.transformers.autolog(log_models=False, log_datasets=False)
        self._run_name = run_name
        self._algorithm = algorithm
        self._task_type = task_type
        self._text_column = text_column
        self._label_column = label_column
        self._hyperparams = hp

    def _start_run(self):
        from mlflow.tracking.context import registry as ctx_registry  # noqa: F401

        return mlflow.start_run(run_name=self._run_name)

    def _log_common_tags(self, strategy: str, n_samples: int) -> None:
        mlflow.set_tags({
            "framework": "huggingface",
            "domain": "nlp",
            "algorithm": self._algorithm,
            "task_type": self._task_type,
            "tenant_id": self.tenant_id,
            "text_column": self._text_column,
            "label_column": self._label_column or "",
            "validation_strategy": strategy,
            "n_samples": str(n_samples),
        })
        mlflow.log_params(self._hyperparams)

    # ──────────────────────────────────────────────────────────────────────
    # Tokenización
    # ──────────────────────────────────────────────────────────────────────

    def _tokenize(
        self, df: pd.DataFrame, text_column: str, label_column: str,
        task_type: str, tokenizer, max_seq_length: int,
    ) -> Any:
        from datasets import Dataset

        if task_type == "token_classification":
            unique_labels = sorted(set(" ".join(df[label_column].astype(str)).split()))
            self.label2id = {lbl: i for i, lbl in enumerate(unique_labels)}
            self.id2label = {i: lbl for lbl, i in self.label2id.items()}

            def _tok(batch):
                texts = [t.split() for t in batch[text_column]]
                labels_batch = [x.split() for x in batch[label_column]]
                tok = tokenizer(
                    texts, is_split_into_words=True,
                    padding="max_length", truncation=True, max_length=max_seq_length,
                )
                aligned = []
                for i, word_labels in enumerate(labels_batch):
                    word_ids = tok.word_ids(batch_index=i)
                    row = []
                    prev = None
                    for wid in word_ids:
                        if wid is None or wid == prev:
                            row.append(-100)
                        else:
                            row.append(
                                self.label2id.get(word_labels[wid], -100)
                                if wid < len(word_labels) else -100
                            )
                        prev = wid
                    aligned.append(row)
                tok["labels"] = aligned
                return tok

            ds = Dataset.from_dict(df.to_dict(orient="list")).map(_tok, batched=True)
            return ds

        # Clasificación / sentimiento
        unique_labels = sorted(df[label_column].unique())
        self.label2id = {lbl: i for i, lbl in enumerate(unique_labels)}
        self.id2label = {i: lbl for lbl, i in self.label2id.items()}

        def _tok(batch):
            tok = tokenizer(
                batch[text_column],
                padding="max_length", truncation=True, max_length=max_seq_length,
            )
            tok["labels"] = [self.label2id[x] for x in batch[label_column]]
            return tok

        ds = Dataset.from_dict(df.to_dict(orient="list")).map(_tok, batched=True)
        ds = ds.remove_columns([text_column, label_column])
        return ds

    # ──────────────────────────────────────────────────────────────────────
    # Métricas
    # ──────────────────────────────────────────────────────────────────────

    def _compute_metrics(self, pred) -> Dict[str, float]:
        from sklearn.metrics import (
            accuracy_score,
            precision_recall_fscore_support,
        )

        predictions = np.argmax(pred.predictions, axis=-1)
        labels = pred.label_ids

        if self._task_type == "token_classification":
            true_pred = [
                [self.id2label[p] for p, lbl in zip(ps, ls) if lbl != -100]
                for ps, ls in zip(predictions, labels)
            ]
            true_lbl = [
                [self.id2label[lbl] for lbl in ls if lbl != -100] for ls in labels
            ]
            from seqeval.metrics import (
                accuracy_score as seq_accuracy,
                classification_report,
                precision_recall_fscore_support as seq_prf,
            )

            precision, recall, f1, _ = seq_prf(
                true_lbl, true_pred, average="weighted", zero_division=0
            )
            report = classification_report(true_lbl, true_pred, output_dict=True)
            return {
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "accuracy": float(seq_accuracy(true_lbl, true_pred)),
                "entity_report": report,
            }

        valid = labels != -100
        acc = accuracy_score(labels[valid], predictions[valid])
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels[valid], predictions[valid], average="weighted", zero_division=0
        )
        return {
            "accuracy": float(acc),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }

    # ──────────────────────────────────────────────────────────────────────
    # Holdout
    # ──────────────────────────────────────────────────────────────────────

    def _train_holdout(
        self, df, text_column, label_column, hf_model, task_type, hp,
        test_size, shuffle, random_state, algorithm,
    ) -> Dict[str, Any]:
        from sklearn.model_selection import train_test_split

        t_start = time.perf_counter()

        train_df, eval_df = train_test_split(
            df, test_size=test_size, random_state=random_state,
            shuffle=shuffle, stratify=df[label_column]
            if task_type != "token_classification" else None,
        )

        tokenizer = AutoTokenizer.from_pretrained(hf_model)
        max_seq_length = int(hp["max_seq_length"])

        train_ds = self._tokenize(train_df, text_column, label_column, task_type, tokenizer, max_seq_length)
        eval_ds = self._tokenize(eval_df, text_column, label_column, task_type, tokenizer, max_seq_length)

        model_cls = (
            AutoModelForTokenClassification if task_type == "token_classification"
            else AutoModelForSequenceClassification
        )
        model = model_cls.from_pretrained(hf_model, num_labels=len(self.label2id))

        out_dir = Path(settings.DATA_DIR) / "tenants" / self.tenant_id / "nlp_models"
        out_dir.mkdir(parents=True, exist_ok=True)

        with self._start_run() as run:
            self._log_common_tags("holdout", len(train_df))
            args = TrainingArguments(
                output_dir=str(out_dir / run.info.run_id[:8]),
                num_train_epochs=int(hp["epochs"]),
                learning_rate=float(hp["learning_rate"]),
                per_device_train_batch_size=int(hp["batch_size"]),
                per_device_eval_batch_size=int(hp["batch_size"]),
                weight_decay=float(hp["weight_decay"]),
                warmup_ratio=float(hp["warmup_ratio"]),
                eval_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                save_total_limit=1,
                report_to=[],
            )
            trainer = Trainer(
                model=model,
                args=args,
                train_dataset=train_ds,
                eval_dataset=eval_ds,
                tokenizer=tokenizer,
                compute_metrics=self._compute_metrics,
            )
            trainer.train()
            metrics = trainer.evaluate()

            # Guardar modelo + tokenizer
            model_dir = out_dir / f"model_{run.info.run_id[:8]}"
            tokenizer_dir = out_dir / f"tokenizer_{run.info.run_id[:8]}"
            trainer.save_model(str(model_dir))
            tokenizer.save_pretrained(str(tokenizer_dir))

            # Export ONNX (best-effort; fallback to saved transformers checkpoint)
            onnx_dir = out_dir / f"onnx_{run.info.run_id[:8]}"
            onnx_path = None
            try:
                onnx_path = export_model_to_onnx(
                    model, tokenizer, onnx_dir, max_seq_length=int(hp["max_seq_length"])
                )
                mlflow.log_artifact(str(onnx_path), artifact_path="onnx")
            except Exception as exc:
                logger.warning("Exportación ONNX omitida (usando checkpoint transformers): %s", exc)
            mlflow.log_artifacts(str(model_dir), artifact_path="model")
            mlflow.log_artifacts(str(tokenizer_dir), artifact_path="tokenizer")

            run_id = run.info.run_id

        self._cleanup_predictions_objects(trainer, eval_df)

        result_metrics = {
            k: v for k, v in metrics.items() if not isinstance(v, dict)
        }
        if "entity_report" in metrics:
            result_metrics["entity_report"] = metrics["entity_report"]

        return {
            "mlflow_run_id": run_id,
            "model_path": str(model_dir),
            "tokenizer_path": str(tokenizer_dir),
            "onnx_path": str(onnx_path) if onnx_path else None,
            "label_map": self.label2id,
            "id2label": self.id2label,
            "metrics": result_metrics,
            "validation_strategy": "holdout",
            "duration_s": time.perf_counter() - t_start,
        }

    def _cleanup_predictions_objects(self, trainer, eval_df) -> None:
        """Libera memoria de los datasets/objects del Trainer tras entrenar."""
        try:
            import gc

            del trainer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    # ──────────────────────────────────────────────────────────────────────
    # Cross-validation
    # ──────────────────────────────────────────────────────────────────────

    def _train_cv(
        self, df, text_column, label_column, hf_model, task_type, hp,
        n_folds, shuffle, random_state, algorithm,
    ) -> Dict[str, Any]:
        from sklearn.model_selection import StratifiedKFold

        t_start = time.perf_counter()
        tokenizer = AutoTokenizer.from_pretrained(hf_model)
        max_seq_length = int(hp["max_seq_length"])

        if task_type == "token_classification":
            cv = StratifiedKFold(
                n_splits=n_folds, shuffle=shuffle, random_state=random_state
            )
            groups = df[label_column].apply(lambda x: x.split()[0])
        else:
            cv = StratifiedKFold(
                n_splits=n_folds, shuffle=shuffle, random_state=random_state
            )
            groups = df[label_column]

        fold_metrics: List[Dict[str, float]] = []
        model_cls = (
            AutoModelForTokenClassification if task_type == "token_classification"
            else AutoModelForSequenceClassification
        )

        out_dir = Path(settings.DATA_DIR) / "tenants" / self.tenant_id / "nlp_models"
        out_dir.mkdir(parents=True, exist_ok=True)

        with self._start_run() as run:
            self._log_common_tags("cross_validation", len(df))
            for fold_idx, (tr_idx, va_idx) in enumerate(cv.split(df, groups)):
                train_df = df.iloc[tr_idx]
                eval_df = df.iloc[va_idx]
                train_ds = self._tokenize(train_df, text_column, label_column, task_type, tokenizer, max_seq_length)
                eval_ds = self._tokenize(eval_df, text_column, label_column, task_type, tokenizer, max_seq_length)

                model = model_cls.from_pretrained(hf_model, num_labels=len(self.label2id))
                args = TrainingArguments(
                    output_dir=str(out_dir / f"cv{fold_idx}_{run.info.run_id[:6]}"),
                    num_train_epochs=int(hp["epochs"]),
                    learning_rate=float(hp["learning_rate"]),
                    per_device_train_batch_size=int(hp["batch_size"]),
                    per_device_eval_batch_size=int(hp["batch_size"]),
                    weight_decay=float(hp["weight_decay"]),
                    warmup_ratio=float(hp["warmup_ratio"]),
                    eval_strategy="epoch",
                    save_strategy="no",
                    report_to=[],
                )
                trainer = Trainer(
                    model=model, args=args, train_dataset=train_ds,
                    eval_dataset=eval_ds, tokenizer=tokenizer,
                    compute_metrics=self._compute_metrics,
                )
                trainer.train()
                m = trainer.evaluate()
                numeric = {k: float(v) for k, v in m.items() if isinstance(v, (int, float))}
                fold_metrics.append(numeric)
                for k, v in numeric.items():
                    mlflow.log_metric(f"fold_{fold_idx}_{k}", v)
                self._cleanup_predictions_objects(trainer, eval_df)

            # Modelo final sobre todos los datos
            full_ds = self._tokenize(df, text_column, label_column, task_type, tokenizer, max_seq_length)
            model = model_cls.from_pretrained(hf_model, num_labels=len(self.label2id))
            args = TrainingArguments(
                output_dir=str(out_dir / f"final_{run.info.run_id[:6]}"),
                num_train_epochs=int(hp["epochs"]),
                learning_rate=float(hp["learning_rate"]),
                per_device_train_batch_size=int(hp["batch_size"]),
                weight_decay=float(hp["weight_decay"]),
                warmup_ratio=float(hp["warmup_ratio"]),
                report_to=[],
            )
            trainer = Trainer(model=model, args=args, train_dataset=full_ds, tokenizer=tokenizer)
            trainer.train()

            model_dir = out_dir / f"model_{run.info.run_id[:8]}"
            tokenizer_dir = out_dir / f"tokenizer_{run.info.run_id[:8]}"
            trainer.save_model(str(model_dir))
            tokenizer.save_pretrained(str(tokenizer_dir))
            onnx_dir = out_dir / f"onnx_{run.info.run_id[:8]}"
            onnx_path = export_model_to_onnx(
                model, tokenizer, onnx_dir, max_seq_length=max_seq_length
            )
            mlflow.log_artifact(str(onnx_path), artifact_path="onnx")
            mlflow.log_artifacts(str(model_dir), artifact_path="model")
            mlflow.log_artifacts(str(tokenizer_dir), artifact_path="tokenizer")
            run_id = run.info.run_id
            self._cleanup_predictions_objects(trainer, df)

        metrics: Dict[str, Any] = {}
        all_keys = fold_metrics[0].keys()
        for k in all_keys:
            values = [fm[k] for fm in fold_metrics]
            metrics[f"cv_mean_{k}"] = float(np.mean(values))
            metrics[f"cv_std_{k}"] = float(np.std(values))

        return {
            "mlflow_run_id": run_id,
            "model_path": str(model_dir),
            "tokenizer_path": str(tokenizer_dir),
            "onnx_path": str(onnx_path) if onnx_path else None,
            "label_map": self.label2id,
            "id2label": self.id2label,
            "metrics": metrics,
            "validation_strategy": "cross_validation",
            "n_folds": n_folds,
            "duration_s": time.perf_counter() - t_start,
        }
