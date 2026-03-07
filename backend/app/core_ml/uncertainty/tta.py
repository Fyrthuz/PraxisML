import math
import random
import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import Tuple

from app.core_ml.uncertainty.base import BaseUncertaintyEstimator

EPS = 1e-6

class RandomImageTransformer:
    def __init__(self, degrees=(-30, 30), translate=(0.1, 0.1), scale=(0.9, 1.1), shear=(-10, 10),
                 padding_mode='border'):
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.padding_mode = padding_mode

    def _get_forward_affine_matrix(self, center, angle, translate, scale, shear):
        cx, cy = center
        tx, ty = translate
        angle_rad = math.radians(angle)
        shear_rad = math.radians(shear)

        R = torch.tensor([
            [math.cos(angle_rad), -math.sin(angle_rad), 0],
            [math.sin(angle_rad),  math.cos(angle_rad), 0],
            [0,                   0,                  1]
        ])

        S = torch.tensor([
            [1, math.tan(shear_rad), 0],
            [0, 1,                   0],
            [0, 0,                   1]
        ])

        Sc = torch.tensor([
            [scale, 0,     0],
            [0,     scale, 0],
            [0,     0,     1]
        ])

        A = R @ S @ Sc

        T_neg = torch.tensor([
            [1, 0, -cx],
            [0, 1, -cy],
            [0, 0, 1]
        ])
        T_pos = torch.tensor([
            [1, 0, cx + tx],
            [0, 1, cy + ty],
            [0, 0, 1]
        ])

        M = T_pos @ A @ T_neg
        return M

    @staticmethod
    def _convert_affine_matrix_to_theta(M, width, height, device):
        T_denorm = torch.tensor([
            [width / 2.0, 0,           width / 2.0],
            [0,           height / 2.0, height / 2.0],
            [0,           0,           1]
        ], dtype=torch.float32, device=device)

        T_norm = torch.tensor([
            [2.0 / width, 0,          -1],
            [0,          2.0 / height, -1],
            [0,          0,           1]
        ], dtype=torch.float32, device=device)

        M = M.to(device).float()

        theta = T_norm @ M @ T_denorm
        return theta[:2, :]

    def transform(self, image):
        device = image.device
        C, H, W = image.shape[-3:] # Assume single image or batched [B, C, H, W]
        center = (W / 2.0, H / 2.0)

        angle = random.uniform(self.degrees[0], self.degrees[1])
        max_dx = self.translate[0] * W
        max_dy = self.translate[1] * H
        tx = random.uniform(-max_dx, max_dx)
        ty = random.uniform(-max_dy, max_dy)
        scale_factor = random.uniform(self.scale[0], self.scale[1])
        shear_angle = random.uniform(self.shear[0], self.shear[1])

        M = self._get_forward_affine_matrix(center, angle, (tx, ty), scale_factor, shear_angle)
        M_inv = torch.inverse(M)

        theta = self._convert_affine_matrix_to_theta(M, W, H, device)

        if image.dim() == 3:
            image_batch = image.unsqueeze(0)
        else:
            image_batch = image

        grid = F.affine_grid(theta.unsqueeze(0).repeat(image_batch.size(0), 1, 1), image_batch.size(), align_corners=False)
        transformed = F.grid_sample(image_batch, grid, align_corners=False, padding_mode=self.padding_mode)

        if image.dim() == 3:
            transformed = transformed.squeeze(0)

        return transformed, M_inv

    def invert(self, transformed_image, M_inv):
        device = transformed_image.device
        C, H, W = transformed_image.shape[-3:]

        theta_inv = self._convert_affine_matrix_to_theta(M_inv, W, H, device)

        if transformed_image.dim() == 3:
            image_batch = transformed_image.unsqueeze(0)
        else:
            image_batch = transformed_image

        grid_inv = F.affine_grid(theta_inv.unsqueeze(0).repeat(image_batch.size(0), 1, 1), image_batch.size(), align_corners=False)
        restored = F.grid_sample(image_batch, grid_inv, align_corners=False, padding_mode=self.padding_mode)

        if transformed_image.dim() == 3:
            restored = restored.squeeze(0)

        return restored


class TTAEstimator(BaseUncertaintyEstimator):
    def __init__(self, model: nn.Module, device: torch.device, tta_samples: int = 10):
        super().__init__(model, device)
        self.tta_samples = tta_samples
        self.transformer = RandomImageTransformer()

    def compute_uncertainty(self, x: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x.to(self.device)
        self.model.eval()

        tta_probs = []
        with torch.no_grad():
            for _ in range(self.tta_samples):
                # 1. Transform the image
                transformed_x, m_inv = self.transformer.transform(x)

                # 2. Forward pass
                logits = self.model(transformed_x)
                if logits.shape[1] == 1: # Binary segmentation
                    logits = torch.cat([1-logits, logits], dim=1)
                probs = F.softmax(logits, dim=1)

                # 3. Invert transformation on predictions
                restored_probs = self.transformer.invert(probs, m_inv)
                tta_probs.append(restored_probs)

        tta_probs = torch.stack(tta_probs) # (S, B, C, H, W)
        avg_probs = tta_probs.mean(dim=0) # (B, C, H, W)

        # Calculate entropy as uncertainty
        entropy = -torch.sum(avg_probs * torch.log(avg_probs + EPS), dim=1) # (B, H, W)

        return avg_probs, entropy
