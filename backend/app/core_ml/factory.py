from enum import Enum
from typing import Dict, Any, Type, Tuple, List, Optional
import torch
import torch.nn as nn

from app.core_ml.interfaces import IUncertaintyAlgorithm
from app.core_ml.uncertainty import (
    MCDropoutEstimator,
    CalibratedMCDropoutEstimator,
    TTAEstimator,
    NoisyInferenceEstimator,
    EnsembleUncertaintyEstimator,
    SklearnEntropyEstimator,
    TreeVarianceEstimator,
    ConformalEstimator,
)


class UncertaintyMethod(str, Enum):
    # ── PyTorch methods ──
    MC_DROPOUT = "mc_dropout"
    CALIBRATED_MC_DROPOUT = "calibrated_mc_dropout"
    TTA = "tta"
    NOISY_INFERENCE = "noisy_inference"
    ENSEMBLE = "ensemble"
    NONE = "none"                      # Inferencia normal sin incertidumbre

    # ── Sklearn methods ──
    ENTROPY = "entropy"                # Entropía de predict_proba
    TREE_VARIANCE = "tree_variance"    # Varianza entre árboles (RF/Bagging)
    CONFORMAL = "conformal"            # Conformal Prediction


# Métodos disponibles para cada framework
PYTORCH_METHODS = {
    UncertaintyMethod.MC_DROPOUT,
    UncertaintyMethod.CALIBRATED_MC_DROPOUT,
    UncertaintyMethod.TTA,
    UncertaintyMethod.NOISY_INFERENCE,
    UncertaintyMethod.ENSEMBLE,
    UncertaintyMethod.NONE,
}

SKLEARN_METHODS = {
    UncertaintyMethod.ENTROPY,
    UncertaintyMethod.TREE_VARIANCE,
    UncertaintyMethod.CONFORMAL,
    UncertaintyMethod.NONE,
}


class PredictionFactory:
    """
    Fábrica encargada de instanciar el algoritmo de incertidumbre correcto
    basado en el método solicitado por el cliente.
    Soporta tanto modelos PyTorch como Scikit-learn.
    """

    @staticmethod
    def get_estimator(
        method: UncertaintyMethod,
        model: Any,
        device: torch.device = None,
        **kwargs,
    ) -> IUncertaintyAlgorithm:
        """
        Retorna una instancia configurada de IUncertaintyAlgorithm.

        Args:
            method: Método de incertidumbre seleccionado.
            model: Modelo (PyTorch nn.Module o sklearn estimator).
            device: Dispositivo de ejecución (CPU/GPU). Solo para PyTorch.
            **kwargs: Hiperparámetros específicos del método.
        """
        # ── Sklearn estimators (no necesitan device) ─────────────────────────
        if method == UncertaintyMethod.ENTROPY:
            return SklearnEntropyEstimator(model=model)

        elif method == UncertaintyMethod.TREE_VARIANCE:
            return TreeVarianceEstimator(model=model)

        elif method == UncertaintyMethod.CONFORMAL:
            alpha = kwargs.get("alpha", 0.1)
            estimator = ConformalEstimator(model=model, alpha=alpha)
            # Si hay datos de calibración, calibrar
            X_cal = kwargs.get("X_calibration")
            y_cal = kwargs.get("y_calibration")
            if X_cal is not None and y_cal is not None:
                estimator.calibrate(X_cal, y_cal)
            return estimator

        # ── PyTorch estimators (necesitan device) ────────────────────────────
        if device is None:
            device = torch.device("cpu")

        if method == UncertaintyMethod.MC_DROPOUT:
            return MCDropoutEstimator(
                model=model,
                device=device,
                mc_samples=kwargs.get("mc_samples", 10),
                p=kwargs.get("p_dropout", 0.2),
            )

        elif method == UncertaintyMethod.CALIBRATED_MC_DROPOUT:
            raise NotImplementedError(
                "La calibración debe ejecutarse en un worker independiente antes de la inferencia final."
            )

        elif method == UncertaintyMethod.TTA:
            return TTAEstimator(
                model=model,
                device=device,
                tta_samples=kwargs.get("tta_samples", 10),
            )

        elif method == UncertaintyMethod.NOISY_INFERENCE:
            return NoisyInferenceEstimator(
                model=model,
                device=device,
                n_samples=kwargs.get("n_samples", 10),
                noise_std=kwargs.get("noise_std", 0.1),
            )

        elif method == UncertaintyMethod.ENSEMBLE:
            return EnsembleUncertaintyEstimator(
                model=model,
                device=device,
                mc_samples=kwargs.get("mc_samples", 8),
                p_dropout=kwargs.get("p_dropout", 0.2),
                tta_samples=kwargs.get("tta_samples", 8),
                noise_samples=kwargs.get("n_samples", 8),
                noise_std=kwargs.get("noise_std", 0.05),
                weights=kwargs.get("ensemble_weights", None),
                epistemic_weight=kwargs.get("epistemic_weight", 0.5),
            )

        elif method == UncertaintyMethod.NONE:
            # Detectar si es sklearn o pytorch
            if isinstance(model, nn.Module):
                from app.core_ml.uncertainty.base import BaseUncertaintyEstimator

                class NoUncertaintyEstimator(BaseUncertaintyEstimator):
                    def compute_uncertainty(
                        self, x: torch.Tensor, **kw
                    ) -> Tuple[torch.Tensor, torch.Tensor]:
                        x = x.to(self.device)
                        self.model.eval()
                        with torch.no_grad():
                            logits = self.model(x)
                            if logits.shape[1] == 1:
                                logits = torch.cat([1 - logits, logits], dim=1)
                            probs = torch.nn.functional.softmax(logits, dim=1)
                            entropy = torch.zeros(
                                (probs.shape[0], probs.shape[2], probs.shape[3]),
                                device=probs.device,
                            )
                        return probs, entropy

                return NoUncertaintyEstimator(model, device)
            else:
                # Sklearn model: predicción directa sin incertidumbre
                from app.core_ml.uncertainty.sklearn_uncertainty import BaseSklearnEstimator
                import numpy as np

                class NoUncertaintySklearn(BaseSklearnEstimator):
                    def estimate_uncertainty(self, input_data, **kw):
                        prediction = self.model.predict(input_data)
                        return {
                            "prediction": prediction,
                            "uncertainty": np.zeros(len(input_data)),
                        }

                return NoUncertaintySklearn(model)

        raise ValueError(f"Método de incertidumbre no soportado: {method}")

    @staticmethod
    def get_available_methods(framework: str = "pytorch") -> list:
        """Devuelve los métodos de incertidumbre disponibles para un framework."""
        if framework == "sklearn":
            return [m.value for m in SKLEARN_METHODS]
        return [m.value for m in PYTORCH_METHODS]
