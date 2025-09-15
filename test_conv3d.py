import torch
import torch.nn as nn

input = torch.randn((1024, 3, 2, 14, 14))
conv_3d = nn.Conv3d(3, 1280, (2, 14, 14), (2, 14, 14), bias=False)
print(f"conv_3d.weight.shape: {conv_3d.weight.shape}")
out1 = conv_3d.forward(input)
out1 = out1.squeeze(-1).squeeze(-1).squeeze(-1)
print(f"out1.shape: {out1.shape}")
print(f"out1: {out1}")


conv_3d_weight = conv_3d.weight
input = input.reshape((1024, -1))
conv_3d_weight = conv_3d_weight.reshape((1280, -1))
out2 = input @ conv_3d_weight.T
print(f"out2.shape: {out2.shape}")
print(f"out2: {out2}")

dif_sum = torch.sum(torch.abs(out1-out2)).item()
dif_max = torch.max(torch.abs(out1-out2)).item()
print(f"max dif: {dif_max}, sum dif: {dif_sum}")