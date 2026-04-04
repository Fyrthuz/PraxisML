from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from app.core_ml.uncertainty.base import BaseUncertaintyEstimator

EPS = 1e-6

class NoisyInferenceEstimator(BaseUncertaintyEstimator):
    """
    Estimates uncertainty by adding Gaussian noise to the input image multiple times.
    This helps measure the robustness and uncertainty of the model against small perturbations.
    """
    def __init__(self, model: nn.Module, device: torch.device, n_samples: int = 10, noise_std: float = 0.1):
        super().__init__(model, device)
        self.n_samples = n_samples
        self.noise_std = noise_std

    def _add_noise(self, x: torch.Tensor) -> torch.Tensor:
        """Adds Gaussian noise to the input tensor."""
        noise = torch.randn_like(x) * self.noise_std
        noisy_x = x + noise
        noisy_x = torch.clamp(noisy_x, 0, 1)  # Assuming image inputs are normalized to [0, 1]
        return noisy_x

    def compute_uncertainty(self, x: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x.to(self.device)
        self.model.eval()

        noisy_probs = []
        with torch.no_grad():
            for _ in range(self.n_samples):
                noisy_x = self._add_noise(x)

                logits = self.model(noisy_x)
                if logits.shape[1] == 1: # Binary segmentation support
                    logits = torch.cat([1-logits, logits], dim=1)

                probs = F.softmax(logits, dim=1)
                noisy_probs.append(probs)

        noisy_probs = torch.stack(noisy_probs) # (S, B, C, H, W)
        avg_probs = noisy_probs.mean(dim=0)    # (B, C, H, W)

        # Calculate entropy as the uncertainty metric
        entropy = -torch.sum(avg_probs * torch.log(avg_probs + EPS), dim=1) # (B, H, W)

        return avg_probs, entropy
