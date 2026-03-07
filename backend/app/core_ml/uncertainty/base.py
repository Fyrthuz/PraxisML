import torch
import torch.nn as nn
from abc import abstractmethod
import numpy as np
from typing import Dict, Tuple

from app.core_ml.interfaces import IUncertaintyAlgorithm

class BaseUncertaintyEstimator(IUncertaintyAlgorithm):
    """
    Base class for uncertainty estimation methods.
    Adapts the pure mathematical interface (`IUncertaintyAlgorithm`) to PyTorch Tensors.
    """
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self.model.to(self.device)

    @abstractmethod
    def compute_uncertainty(self, x: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the prediction and uncertainty given an input tensor or batch.

        Args:
            x (torch.Tensor): Input image tensor of shape (B, C, H, W).
            **kwargs: Additional parameters for specific implementations.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - Average prediction probabilities: (B, NumClasses, H, W)
                - Uncertainty mapping (e.g., entropy or variance): (B, H, W)
        """
        pass

    def estimate_uncertainty(self, input_data: np.ndarray, **kwargs) -> Dict[str, np.ndarray]:
        """
        Implementation of IUncertaintyAlgorithm.
        Handles the transformation from NumPy to Tensor and runs the internal PyTorch methods.
        """
        # Handle pandas dataframe if tabular data
        if hasattr(input_data, "to_numpy"):
            input_data = input_data.to_numpy()

        # Convert NumPy to Tensor and move to device
        tensor_x = torch.from_numpy(input_data).float().to(self.device)

        # Execute the pytorch specific method
        pred_tensor, unc_tensor = self.compute_uncertainty(tensor_x, **kwargs)

        # Convert back to contiguous NumPy arrays to send to backend services
        return {
            "prediction": pred_tensor.detach().cpu().numpy(),
            "uncertainty": unc_tensor.detach().cpu().numpy()
        }
