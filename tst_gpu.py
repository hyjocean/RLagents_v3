import torch

# 确保 CUDA 可用
assert torch.cuda.is_available()

# 选择 CUDA 设备
device = torch.device("cuda:0")

# 创建一些随机数据张量
a = torch.randn(1024, 1024, device=device)
b = torch.randn(1024, 1024, device=device)

# 执行一个 cuBLAS 操作（矩阵乘法）
c = torch.matmul(a, b)

print("cuBLAS operation successful")
