"""
Estimadores de incertidumbre para modelos Scikit-learn.
Implementan `IUncertaintyAlgorithm` al igual que los estimadores de PyTorch,
pero trabajan directamente con arrays NumPy y modelos sklearn.
"""
import logging
from typing import Dict, Any, Optional

import numpy as np

from app.core_ml.interfaces import IUncertaintyAlgorithm

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Base class for sklearn uncertainty estimators
# ──────────────────────────────────────────────────────────────────────────────

class BaseSklearnEstimator(IUncertaintyAlgorithm):
    """
    Base para estimadores de incertidumbre que operan sobre modelos sklearn.
    A diferencia de BaseUncertaintyEstimator (PyTorch), trabaja con ndarray puro.
    """

    def __init__(self, model: Any):
        self.model = model

    def estimate_uncertainty(self, input_data: np.ndarray, **kwargs) -> Dict[str, np.ndarray]:
        """
        Interfaz unificada: recibe ndarray, devuelve dict con 'prediction' y 'uncertainty'.
        """
        raise NotImplementedError


# ──────────────────────────────────────────────────────────────────────────────
# 1. Entropía de predict_proba
# ──────────────────────────────────────────────────────────────────────────────

class SklearnEntropyEstimator(BaseSklearnEstimator):
    """
    Calcula incertidumbre como la entropía de Shannon de las probabilidades
    predichas por cualquier clasificador sklearn con `predict_proba`.

    Entropía alta = el modelo está "indeciso" entre clases.
    Entropía baja = el modelo está seguro.

    Output normalizado a [0, 1] (dividido por log(n_classes)).
    """

    def estimate_uncertainty(self, input_data: np.ndarray, **kwargs) -> Dict[str, np.ndarray]:
        if not hasattr(self.model, "predict_proba"):
            raise ValueError(
                f"El modelo {type(self.model).__name__} no soporta predict_proba. "
                "Usa un clasificador que lo soporte (RandomForest, LogisticRegression, etc.)"
            )

        probs = self.model.predict_proba(input_data)
        prediction = self.model.predict(input_data)

        # Entropía de Shannon: H = -Σ p_i * log(p_i)
        eps = 1e-10
        entropy = -np.sum(probs * np.log(probs + eps), axis=1)

        # Normalizar por log(n_classes) para obtener [0, 1]
        n_classes = probs.shape[1]
        if n_classes > 1:
            entropy = entropy / np.log(n_classes + eps)

        return {
            "prediction": prediction,
            "uncertainty": entropy,
            "probabilities": probs,
        }


# ──────────────────────────────────────────────────────────────────────────────
# 2. Varianza entre árboles (Random Forest / Bagging)
# ──────────────────────────────────────────────────────────────────────────────

class TreeVarianceEstimator(BaseSklearnEstimator):
    """
    Calcula incertidumbre epistémica como el desacuerdo entre árboles individuales
    de un ensemble (RandomForest, GradientBoosting, BaggingClassifier, etc.).

    Para clasificación: varianza de las proporciones de voto.
    Para regresión: varianza de las predicciones individuales.
    """

    def estimate_uncertainty(self, input_data: np.ndarray, **kwargs) -> Dict[str, np.ndarray]:
        if not hasattr(self.model, "estimators_"):
            raise ValueError(
                f"El modelo {type(self.model).__name__} no tiene estimators_. "
                "Usa un ensemble (RandomForest, BaggingClassifier, etc.)"
            )

        prediction = self.model.predict(input_data)

        # Intentar extraer predicciones individuales de cada árbol
        is_classifier = hasattr(self.model, "predict_proba")

        if is_classifier:
            # Predicción de cada árbol
            tree_predictions = np.array([
                estimator.predict(input_data) for estimator in self.model.estimators_
            ])
            # Varianza del voto mayoritario por muestra
            # Para cada muestra, computar la proporción de cada clase y tomar varianza
            n_samples = input_data.shape[0]
            variance = np.zeros(n_samples)

            for i in range(n_samples):
                votes = tree_predictions[:, i]
                unique, counts = np.unique(votes, return_counts=True)
                proportions = counts / len(votes)
                # Varianza de proporciones = medida de desacuerdo
                variance[i] = 1.0 - np.max(proportions)  # 0 = consenso total, ~1 = desacuerdo total
        else:
            # Regresión: varianza directa de predicciones
            tree_predictions = np.array([
                estimator.predict(input_data) for estimator in self.model.estimators_
            ])
            variance = tree_predictions.var(axis=0)

        return {
            "prediction": prediction,
            "uncertainty": variance,
        }


# ──────────────────────────────────────────────────────────────────────────────
# 3. Conformal Prediction (Split Conformal)
# ──────────────────────────────────────────────────────────────────────────────

class ConformalEstimator(BaseSklearnEstimator):
    """
    Conformal Prediction: produce conjuntos de predicción con garantía
    de cobertura 1-α. Un conjunto más grande = más incertidumbre.

    Requiere calibración previa con un dataset de calibración (calibrate()).
    Si no se calibra, usa un fallback heurístico basado en predict_proba.

    La incertidumbre es el tamaño del conjunto de predicción normalizado.
    """

    def __init__(self, model: Any, alpha: float = 0.1):
        super().__init__(model)
        self.alpha = alpha
        self.calibration_scores: Optional[np.ndarray] = None
        self.quantile_threshold: Optional[float] = None

    def calibrate(self, X_cal: np.ndarray, y_cal: np.ndarray) -> None:
        """
        Calibra el estimador con un dataset de calibración.
        Computa los nonconformity scores y el quantil para el nivel alpha.
        """
        if not hasattr(self.model, "predict_proba"):
            raise ValueError("Conformal Prediction requiere predict_proba.")

        probs = self.model.predict_proba(X_cal)
        n = len(y_cal)

        # Nonconformity score: 1 - prob(true class)
        true_class_probs = probs[np.arange(n), y_cal.astype(int)]
        self.calibration_scores = 1.0 - true_class_probs

        # Compute quantile: ceil((n+1) * (1-alpha)) / n
        quantile_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        quantile_level = min(quantile_level, 1.0)
        self.quantile_threshold = np.quantile(self.calibration_scores, quantile_level)

        logger.info(
            "Conformal calibration: n=%d, alpha=%.2f, threshold=%.4f",
            n, self.alpha, self.quantile_threshold,
        )

    def estimate_uncertainty(self, input_data: np.ndarray, **kwargs) -> Dict[str, np.ndarray]:
        prediction = self.model.predict(input_data)

        if not hasattr(self.model, "predict_proba"):
            # Fallback: sin probabilidades, incertidumbre uniforme
            return {
                "prediction": prediction,
                "uncertainty": np.ones(len(input_data)) * 0.5,
                "prediction_sets": None,
            }

        probs = self.model.predict_proba(input_data)

        if self.quantile_threshold is not None:
            # Calibrado: construir conjuntos de predicción
            # Incluir clase c si 1 - P(c) <= threshold, es decir P(c) >= 1 - threshold
            threshold = 1.0 - self.quantile_threshold
            prediction_sets = probs >= threshold

            # Incertidumbre = tamaño del conjunto / número de clases (normalizado [0,1])
            set_sizes = prediction_sets.sum(axis=1).astype(float)
            n_classes = probs.shape[1]
            uncertainty = (set_sizes - 1) / max(n_classes - 1, 1)
        else:
            # No calibrado: usar heurística basada en max_prob
            max_prob = probs.max(axis=1)
            uncertainty = 1.0 - max_prob
            prediction_sets = None

        return {
            "prediction": prediction,
            "uncertainty": uncertainty,
            "prediction_sets": prediction_sets if prediction_sets is not None else None,
            "probabilities": probs,
        }
