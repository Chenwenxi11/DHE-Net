import torch

# 检查 CUDA 是否可用
cuda_available = torch.cuda.is_available()
print(f"CUDA 可用: {cuda_available}")

# 如果可用，检查当前的 CUDA 版本
if cuda_available:
    print(f"CUDA 版本: {torch.version.cuda}")

# 检查当前设备
device = torch.device("cuda" if cuda_available else "cpu")
print(f"使用的设备: {device}")
