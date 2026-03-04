import torch
import torch.nn as nn
import os
import sys
from pathlib import Path

# Mock settings and MLFlow for testing without a real server if needed, 
# but here we try to use the real logic with a local file URI
backend_path = Path(r"c:\Users\ferna\OneDrive\Escritorio\TFM_productivo\backend")
sys.path.append(str(backend_path))

from app.services.mlflow_service import MLFlowService
from app.core_ml.models.unet import UNet
import mlflow

def test_robust_loading():
    svc = MLFlowService()
    device = torch.device("cpu")
    
    # 1. Create a dummy state_dict model
    model = UNet(in_channels=3, num_classes=2)
    state_dict_path = "/tmp/test_unet_state_dict.pth"
    torch.save(model.state_dict(), state_dict_path)
    
    # 2. Register it in MLFlow as a state_dict (to trigger the error case in the original code)
    print("Registering state_dict model in MLFlow...")
    run_id = svc.register_pth_model(
        pth_path=state_dict_path,
        model_name="TestRobustLoading",
        tenant_id="test_tenant",
        architecture="unet",
        num_classes=2
    )
    print(f"Model registered with run_id: {run_id}")
    
    # 3. Try to load it back
    print("Attempting to load model back...")
    try:
        loaded_model = svc.load_model(run_id, device)
        print("SUCCESS: Model loaded successfully!")
        
        # 4. Verify functionality
        dummy_input = torch.randn(1, 3, 224, 224)
        output = loaded_model(dummy_input)
        print(f"Model output shape: {output.shape}")
        assert output.shape == (1, 2, 224, 224)
        print("VERIFIED: Model is functional.")
        
    except Exception as e:
        print(f"FAILURE: Could not load model: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_robust_loading()
