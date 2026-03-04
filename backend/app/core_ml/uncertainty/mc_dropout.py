import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
from tqdm import tqdm

from app.core_ml.uncertainty.base import BaseUncertaintyEstimator

EPS = 1e-6

class MCDropoutHook:
    def __init__(self, model: nn.Module, p: float = 0.2):
        self.model = model
        self.p = p
        self.hooks = []
        self.enabled = False

    def _apply_mask(self, module, input, output):
        if self.enabled and isinstance(output, torch.Tensor):
            mask = (torch.rand_like(output) > self.p).float()
            return output * mask
        return output

    def enable(self, 
               ignore_specific_layers: Optional[List] = None,
               ignore_type_layers: Optional[List] = None,
               layer_types: Optional[List] = None):
        if self.enabled:
            return

        ignore_specific_layers = ignore_specific_layers or []
        ignore_type_layers = ignore_type_layers or []
        layer_types = layer_types or []

        for name, module in self.model.named_modules():
            if name == "":
                continue

            apply_condition = (
                (isinstance(module, tuple(layer_types)) or not layer_types) and
                module not in ignore_specific_layers and
                not isinstance(module, tuple(ignore_type_layers))
            )

            if apply_condition:
                self.hooks.append(module.register_forward_hook(self._apply_mask))

        self.enabled = True

    def remove(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.enabled = False

    def __enter__(self):
        self.enable()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove()

class MCDropoutEstimator(BaseUncertaintyEstimator):
    def __init__(self, model: nn.Module, device: torch.device, mc_samples: int = 10, p: float = 0.2):
        super().__init__(model, device)
        self.mc_samples = mc_samples
        self.p = p
        self.mc_manager = MCDropoutHook(self.model, p=self.p)

    def compute_uncertainty(self, x: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x.to(self.device)
        self.model.eval()

        self.mc_manager.p = self.p
        self.mc_manager.enable(ignore_type_layers=[nn.ReLU, nn.Softmax])

        mc_probs = []
        with torch.no_grad():
            for _ in range(self.mc_samples):
                logits = self.model(x)
                if logits.shape[1] == 1: # Binary segmentation
                    logits = torch.cat([1-logits, logits], dim=1)
                probs = F.softmax(logits, dim=1)
                mc_probs.append(probs)

        self.mc_manager.remove()

        mc_probs = torch.stack(mc_probs) # (S, B, C, H, W)
        avg_probs = mc_probs.mean(dim=0) # (B, C, H, W)

        # Calculate entropy
        entropy = -torch.sum(avg_probs * torch.log(avg_probs + EPS), dim=1) # (B, H, W)

        return avg_probs, entropy

class CalibratedMCDropoutEstimator(BaseUncertaintyEstimator):
    def __init__(self, model: nn.Module, device: torch.device, data_loader: torch.utils.data.DataLoader, 
                 p_values: List[float] = [0.1, 0.3, 0.5], mc_samples: int = 5, num_classes: int = 2, 
                 calib_tolerance: float = 0.02, scale_entropy: bool = False):
        super().__init__(model, device)
        self.data_loader = data_loader
        self.p_values = p_values
        self.mc_samples = mc_samples
        self.num_classes = num_classes
        self.calib_tolerance = calib_tolerance
        self.scale_entropy = scale_entropy

        self.best_phi = None
        self.best_scale = None
        self.best_scale_relaxed = None
        self.mc_manager = MCDropoutHook(self.model, p=0.0)

    def optimize_parameters(self) -> Tuple[float, float, float]:
        best_phi, best_scale, best_scale_relaxed = None, None, None
        best_loss = float('inf')

        self.model.eval()

        for p in self.p_values:
            self.mc_manager.p = p
            self.mc_manager.enable(ignore_type_layers=[nn.ReLU, nn.Softmax])
            
            total_loss = 0.0
            total_batches = 0

            with torch.no_grad():
                for x, y in tqdm(self.data_loader, desc=f"Testing p={p}"):
                    x, y = x.to(self.device), y.to(self.device)
                    y_indices = y.squeeze(1).long() if y.dim() > 3 else y.long()
                    
                    mc_probs = []
                    for _ in range(self.mc_samples):
                        logits = self.model(x)
                        if logits.shape[1] == 1:
                            logits = torch.cat([1-logits, logits], dim=1)
                        probs = F.softmax(logits, dim=1)
                        mc_probs.append(probs)
                    avg_probs = torch.stack(mc_probs).mean(dim=0)
                    
                    avg_log_probs = torch.log(avg_probs + EPS)
                    loss = F.nll_loss(avg_log_probs, y_indices)
                    total_loss += loss.item()
                    total_batches += 1

                    # Calibration parameters logic...
                    preds = torch.argmax(avg_probs, dim=1)
                    entropy = -torch.sum(avg_probs * torch.log(avg_probs + EPS), dim=1)

                    if self.scale_entropy:
                        entropy = entropy / torch.log(torch.tensor(self.num_classes, dtype=entropy.dtype, device=entropy.device))

                    incorrect = (preds != y_indices).float()
                    entropy_flat = entropy.flatten()
                    incorrect_flat = incorrect.flatten()
                    if entropy_flat.numel() < 2:
                        cov = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
                    else:
                        cov = torch.cov(torch.stack([entropy_flat, incorrect_flat]))
                    scale = 1 / (cov[0, 1] + EPS)
                    scale = scale.item()

                    scale_relaxed = scale
                    if self.calib_tolerance > 0:
                        scale_relaxed = self._find_relaxed_scale(avg_probs, entropy, y_indices, scale)

            avg_loss = total_loss / total_batches
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_phi = p
                best_scale = scale
                best_scale_relaxed = scale_relaxed
                
            self.mc_manager.remove()

        self.best_phi = best_phi
        self.best_scale = best_scale
        self.best_scale_relaxed = best_scale_relaxed
        return best_phi, best_scale, best_scale_relaxed

    def _find_relaxed_scale(self, probs: torch.Tensor, entropy: torch.Tensor, targets: torch.Tensor, init_scale: float) -> float:
        current_scale = init_scale
        best_ece = float('inf')
        best_scale = init_scale
        for _ in range(10):
            scaled_uncertainty = entropy * current_scale
            ece = self._compute_ece(probs, scaled_uncertainty, targets)
            if ece < best_ece:
                best_ece = ece
                best_scale = current_scale
            current_scale *= 0.95 if ece > self.calib_tolerance else 1.05
        return best_scale

    def _compute_ece(self, probs: torch.Tensor, uncertainty: torch.Tensor, targets: torch.Tensor, n_bins: int = 10) -> float:
        bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=probs.device)
        bin_indices = torch.bucketize(uncertainty, bin_boundaries)
        ece = 0.0
        probs_flat = probs.permute(0, 2, 3, 1).reshape(-1, self.num_classes)
        targets_flat = targets.flatten()

        for bin_idx in range(1, n_bins + 1):
            in_bin = (bin_indices == bin_idx)
            if in_bin.any():
                in_bin_flat = in_bin.flatten()
                bin_conf = (1 - uncertainty[in_bin]).mean()
                bin_acc = (torch.argmax(probs_flat[in_bin_flat], dim=1) == targets_flat[in_bin_flat]).float().mean()
                bin_weight = in_bin_flat.float().mean()
                ece += torch.abs(bin_acc - bin_conf) * bin_weight
        return ece.item() if isinstance(ece, torch.Tensor) else ece

    def compute_uncertainty(self, x: torch.Tensor, use_relaxed: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.best_scale is None:
            raise RuntimeError("Call optimize_parameters() first.")

        scale = self.best_scale_relaxed if use_relaxed else self.best_scale
        
        self.mc_manager.p = self.best_phi
        self.mc_manager.enable(ignore_type_layers=[nn.ReLU, nn.Softmax])

        x = x.to(self.device)
        self.model.eval()

        mc_probs = []
        with torch.no_grad():
            for _ in range(self.mc_samples):
                logits = self.model(x)
                if logits.shape[1] == 1:
                    logits = torch.cat([1-logits, logits], dim=1)
                mc_probs.append(F.softmax(logits, dim=1))
        
        self.mc_manager.remove()

        avg_probs = torch.stack(mc_probs).mean(dim=0)
        entropy = -torch.sum(avg_probs * torch.log(avg_probs + EPS), dim=1)
        scaled_uncertainty = entropy * scale

        return avg_probs, scaled_uncertainty
