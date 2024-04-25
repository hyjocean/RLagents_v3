# import torch

# # 确保 CUDA 可用
# assert torch.cuda.is_available()

# # 选择 CUDA 设备
# device = torch.device("cuda:0")

# # 创建一些随机数据张量
# a = torch.randn(1024, 1024, device=device)
# b = torch.randn(1024, 1024, device=device)

# # 执行一个 cuBLAS 操作（矩阵乘法）
# c = torch.matmul(a, b)

# print("cuBLAS operation successful")


# import torch
# import torch.nn as nn
# import numpy as np
# from utils.map2nxG import nx_generate_path

# # 检查是否有可用的 CUDA 设备
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")

# # 创建一个简单的卷积神经网络模型
# class SimpleCNN(nn.Module):
#     def __init__(self):
#         super(SimpleCNN, self).__init__()
#         # 创建一个卷积层
#         # 参数分别为：输入通道数，输出通道数，卷积核大小
#         self.conv1 = nn.Conv2d(in_channels=4, out_channels=512//4, kernel_size=3, stride=1, padding="same")
    
#     def forward(self, x):
#         return self.conv1(x)



# world = np.array([
#     [0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0],
#     [0, 0, 1, 0, 0],
#     [0, 0, 0, 0, 0]
# ])
# path = nx_generate_path(world, [(0, 0), [1,2]], [(4, 4),[1,4]])



# # 实例化模型并移动到 CUDA 设备（如果可用）
# model = SimpleCNN().to(device)

# # 创建一个随机数据 tensor 来模拟一个单通道图像批次
# # 参数分别为：批次大小，通道数，高度，宽度
# map =torch.rand((4,4,10,10)).to(device)

# # 将输入数据传递给模型，执行前向传递
# output = model(map)

# # 打印输出的大小
# print(f"Output size: {output.size()}")

from utils.utils import load_config

a = load_config("/home/bld/HK_RL/RLagents_v3/libMultiRobotPlanning/example2.yaml")
print(a)