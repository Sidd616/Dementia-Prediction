"""Standalone CUDA/torch environment check.

Not used by the training/GUI pipeline (which is pure scikit-learn/xgboost) -
this is a dev utility for confirming a GPU-enabled torch install. Requires
the optional `torch` dependency (see requirements-dev.txt).
Run directly: `python scripts/check_gpu.py`.
"""
import torch

print(f"Is CUDA supported by this system? {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    cuda_id = torch.cuda.current_device()
    print(f"ID of current CUDA device: {cuda_id}")
    print(f"Name of current CUDA device: {torch.cuda.get_device_name(cuda_id)}")
else:
    print("No CUDA device available.")
