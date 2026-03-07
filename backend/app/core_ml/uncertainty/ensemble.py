"""
ensemble.py — EnsembleUncertaintyEstimator

Combina MC Dropout, TTA y Noisy Inference para obtener una estimación de
incertidumbre más robusta y calibrada.

Estrategia:
  1. Cada método produce su mapa de probabilidades (B, C, H, W) y su mapa de
     entropía (B, H, W).
  2. Las probabilidades se promedian con pesos configurables → predicción final.
  3. La incertidumbre final es un promedio ponderado de las entropías de cada método
     más la varianza entre las predicciones de los métodos (desacuerdo epistémico).
"""
import torch
import torch.nn as nn
from typing import Tuple, Dict, List, Optional

from app.core_ml.uncertainty.base import BaseUncertaintyEstimator
from app.core_ml.uncertainty.mc_dropout import MCDropoutEstimator
from app.core_ml.uncertainty.tta import TTAEstimator
from app.core_ml.uncertainty.noise_inference import NoisyInferenceEstimator

EPS = 1e-6


class EnsembleUncertaintyEstimator(BaseUncertaintyEstimator):
    """
    Estimador ensemble que fusiona MC-Dropout, TTA y Noisy Inference.

    Args:
        model: Modelo PyTorch cargado.
        device: CPU o CUDA.
        mc_samples: Muestras para MC Dropout.
        p_dropout: Probabilidad de dropout para MC Dropout.
        tta_samples: Augmentaciones para TTA.
        noise_samples: Muestras ruidosas para Noisy Inference.
        noise_std: Desviación estándar del ruido gaussiano.
        weights: Pesos [w_mc, w_tta, w_noise] para el promedio ponderado.
                 Si None se usan pesos iguales.
        epistemic_weight: Peso (0–1) de la varianza epistémica en la
                          incertidumbre final vs. promedio de entropías.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        mc_samples: int = 8,
        p_dropout: float = 0.2,
        tta_samples: int = 8,
        noise_samples: int = 8,
        noise_std: float = 0.05,
        weights: Optional[List[float]] = None,
        epistemic_weight: float = 0.5,
    ):
        super().__init__(model, device)

        self.epistemic_weight = epistemic_weight

        # Pesos normalizados entre los tres brazos
        raw_w = weights if weights is not None else [1.0, 1.0, 1.0]
        total = sum(raw_w)
        self.weights = [w / total for w in raw_w]

        # Los tres estimadores comparten el mismo model (ya en device)
        self.mc_estimator = MCDropoutEstimator(
            model=model, device=device, mc_samples=mc_samples, p=p_dropout
        )
        self.tta_estimator = TTAEstimator(
            model=model, device=device, tta_samples=tta_samples
        )
        self.noise_estimator = NoisyInferenceEstimator(
            model=model, device=device, n_samples=noise_samples, noise_std=noise_std
        )

        # Exponer hiperparámetros para el logging de MLFlow
        self.mc_samples = mc_samples
        self.p = p_dropout
        self.tta_samples = tta_samples
        self.n_samples = noise_samples
        self.noise_std = noise_std

    def compute_uncertainty(
        self, x: torch.Tensor, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Ejecuta los tres estimadores y fusiona sus salidas.

        Returns:
            avg_probs  (B, C, H, W): predicción de segmentación combinada.
            uncertainty (B, H, W):  mapa de incertidumbre combinado.
        """
        x = x.to(self.device)

        # ── Ejecutar los tres brazos ──────────────────────────────────────────
        mc_probs, mc_entropy = self.mc_estimator.compute_uncertainty(x)
        tta_probs, tta_entropy = self.tta_estimator.compute_uncertainty(x)
        noise_probs, noise_entropy = self.noise_estimator.compute_uncertainty(x)

        w_mc, w_tta, w_noise = self.weights

        # ── 1. Predicción combinada: promedio ponderado de probabilidades ─────
        avg_probs = (
            w_mc * mc_probs
            + w_tta * tta_probs
            + w_noise * noise_probs
        )  # (B, C, H, W)

        # ── 2. Incertidumbre aleatórica: promedio ponderado de entropías ──────
        aleatoric = (
            w_mc * mc_entropy
            + w_tta * tta_entropy
            + w_noise * noise_entropy
        )  # (B, H, W)

        # ── 3. Incertidumbre epistémica: varianza entre las predicciones ───────
        # Apilamos las probabilidades y calculamos la varianza media por píxel
        stacked = torch.stack([mc_probs, tta_probs, noise_probs], dim=0)  # (3, B, C, H, W)
        # Varianza sobre los métodos, promediada sobre clases → (B, H, W)
        epistemic = stacked.var(dim=0).mean(dim=1)  # (B, H, W)

        # ── 4. Incertidumbre final: mezcla de aleatórica y epistémica ─────────
        a = self.epistemic_weight
        uncertainty = (1 - a) * aleatoric + a * epistemic  # (B, H, W)

        return avg_probs, uncertainty

    def estimate_uncertainty(self, input_data, **kwargs) -> Dict[str, object]:
        """Override para devolver también las contribuciones individuales."""

        tensor_x = torch.from_numpy(input_data).float().to(self.device)
        avg_probs, uncertainty = self.compute_uncertainty(tensor_x, **kwargs)

        return {
            "prediction": avg_probs.detach().cpu().numpy(),
            "uncertainty": uncertainty.detach().cpu().numpy(),
            # Desglose por método (útil para inspección / logging)
            "ensemble_weights": {
                "mc_dropout": self.weights[0],
                "tta": self.weights[1],
                "noisy_inference": self.weights[2],
            },
            "epistemic_weight": self.epistemic_weight,
        }
