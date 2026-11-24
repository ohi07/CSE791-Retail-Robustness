"""Run this from your existing Products10K virtual environment."""
import platform
import torch
import torchvision
print('Python:', platform.python_version())
print('PyTorch:', torch.__version__)
print('Torchvision:', torchvision.__version__)
print('CUDA runtime in PyTorch:', torch.version.cuda)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
    tensor = torch.randn(128, 128, device='cuda')
    print('GPU test matrix:', (tensor @ tensor).shape)
    print('GPU setup: READY')
else:
    print('GPU setup: NOT READY. Run nvidia-smi and use the official Windows CUDA/Pip installer.')
